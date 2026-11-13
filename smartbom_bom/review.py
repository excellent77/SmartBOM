"""人工覆核連通塊的應用服務。

本模組只處理「Reducer 邊界分段、風險排序、人工修正與 BOM 重建」，
不依賴 Gradio 或 Plotly，因此可由其他介面與單元測試重用。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import SMALL_SEGMENT_GROUP_THRESHOLD
from .models import BomAnalysisResult, EntityGroup
from .grouping import _components_for_nodes
from .predicates import (
    canonical_name, has_fixed_quarter_inch_size, is_gauge, is_hose,
    is_size_boundary, is_tube,
)
from .reporting import BomReportBuilder
from .text import FINISH_MATERIAL_NAMES, _extract_size, _normalize_size, _tube_description


SEGMENT_PALETTE = (
    "#2563EB", "#7C3AED", "#0891B2", "#059669", "#65A30D", "#CA8A04",
    "#EA580C", "#DC2626", "#DB2777", "#4F46E5", "#0F766E", "#9333EA",
)
COLOR_NAME_RE = re.compile(r"顏色_\d+$")
MATERIAL_RE = re.compile(
    r"(?P<base>SUS\s*\d*[A-Z]?)(?:\s+(?P<finish>AP|EP|BA))?",
    re.IGNORECASE,
)
LENGTH_NAMES = {"tube", "coil tube", "hose", "軟管"}


@dataclass(slots=True)
class ReviewSegment:
    """一個移除 Reducer 邊界後的可覆核連通塊。"""

    segment_id: int
    group_ids: list[int]
    size: str
    specification: str
    component_count: int
    mismatch_count: int
    component_summary: str
    color: str
    order: int = 0
    # 群組編號 → 差異說明；供互動圖高亮及 Hover 顯示。
    mismatches: dict[int, list[str]] = field(default_factory=dict)

    @property
    def code(self) -> str:
        return f"C{self.segment_id + 1:02d}"

    @property
    def is_small(self) -> bool:
        return self.component_count <= SMALL_SEGMENT_GROUP_THRESHOLD

    @property
    def risk_label(self) -> str:
        mismatch = (
            f"🔴 {self.mismatch_count} 項差異"
            if self.mismatch_count >= 3
            else f"🟠 {self.mismatch_count} 項差異"
            if self.mismatch_count
            else "🟢 無差異"
        )
        return f"{mismatch}{'｜◆ 小型連通塊' if self.is_small else ''}"


@dataclass(slots=True)
class ReviewSession:
    """一次上傳檔案的人工覆核狀態。"""

    source: Path
    vectors: pd.DataFrame
    result: BomAnalysisResult
    segments: list[ReviewSegment] = field(default_factory=list)


def _majority(values: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    first = {value: cleaned.index(value) for value in counts}
    return min(counts, key=lambda value: (-counts[value], first[value]))


def _text_evidence(text: str) -> tuple[str, str]:
    """從配對文字擷取可驗證的尺寸與材質；無法擷取的欄位留空。"""
    tube = _tube_description(text)
    if tube:
        return tube["size"], tube["material"]
    size = _extract_size(text)
    material_match = MATERIAL_RE.search(str(text))
    if not material_match:
        return size, ""
    base = re.sub(r"\s+", "", material_match.group("base")).upper()
    finish = (material_match.group("finish") or "").upper()
    return size, f"{base} {finish}".strip()


def _reviewable(group: EntityGroup) -> bool:
    return group.name != "IGNORE" and not COLOR_NAME_RE.fullmatch(str(group.name).strip())


def _segment_defaults(groups: list[EntityGroup]) -> tuple[str, str]:
    tubes = [group for group in groups if is_tube(group)]
    size_values = [group.size for group in tubes if group.size]
    if not size_values:
        size_values = [
            group.size for group in groups
            if group.size and not is_gauge(group) and not is_hose(group)
        ]
    material_values = [group.material for group in tubes if group.material]
    if not material_values:
        material_values = [
            group.material for group in groups
            if group.material and canonical_name(group.name) in FINISH_MATERIAL_NAMES
        ]
    if not material_values:
        material_values = [group.material for group in groups if group.material]
    return _majority(size_values), _majority(material_values)


def _group_mismatches(group: EntityGroup) -> list[str]:
    """比較元件配對文字與其最終欄位，回傳每項差異的說明。

    實際比較採用元件最終欄位，避免 Gauge 與 Hose 等合法特殊規則被誤報。
    """
    if not group.matched_text:
        return []
    evidence_size, evidence_material = _text_evidence(group.matched_text)
    target_size = _normalize_size(group.size) if group.size else ""
    target_material = re.sub(r"\s+", " ", group.material.strip()).upper()
    issues = []
    if evidence_size and target_size and _normalize_size(evidence_size) != target_size:
        issues.append(f"尺寸：文字 {_normalize_size(evidence_size)} ≠ 判定 {target_size}")
    if evidence_material and target_material and evidence_material.upper() != target_material:
        issues.append(f"材質：文字 {evidence_material.upper()} ≠ 判定 {target_material}")
    return issues


def _segment_mismatches(groups: list[EntityGroup]) -> dict[int, list[str]]:
    """回傳連通塊內有差異的元件及其差異說明。"""
    return {
        group.group_id: issues
        for group in groups
        if (issues := _group_mismatches(group))
    }


def _dimension_key(value: str) -> tuple[int, float, str]:
    """建立 Reducer 尺寸排序鍵；同單位由大至小，確保反向輸入得到同一格式。"""
    normalized = _normalize_size(value)
    quote = re.fullmatch(r"(\d+)(?:/(\d+))?\"", normalized)
    if quote:
        number = Fraction(int(quote.group(1)), int(quote.group(2) or 1))
        return 3, float(number), normalized
    nominal = re.fullmatch(r"(\d+(?:\.\d+)?)A", normalized, re.IGNORECASE)
    if nominal:
        return 2, float(nominal.group(1)), normalized
    millimeter = re.fullmatch(r"(\d+(?:\.\d+)?)mm", normalized, re.IGNORECASE)
    if millimeter:
        return 1, float(millimeter.group(1)), normalized
    return 0, 0.0, normalized


def canonical_reducer_size(values: Sequence[str]) -> str:
    """只在 Reducer 所有方向剛好得到兩種尺寸時組合。"""
    unique = list(dict.fromkeys(_normalize_size(value) for value in values if value))
    if len(unique) != 2:
        return ""
    ordered = sorted(unique, key=_dimension_key, reverse=True)
    return "x".join(ordered)


class ConnectivityReviewService:
    """建立覆核資料、套用人工修正並重建最終 BOM。"""

    def __init__(self, report_builder: BomReportBuilder | None = None):
        self.report_builder = report_builder or BomReportBuilder()

    def create_session(
        self,
        source: str | Path,
        vectors: pd.DataFrame,
        result: BomAnalysisResult,
    ) -> ReviewSession:
        session = ReviewSession(Path(source), vectors, result)
        session.segments = self._build_segments(result.groups, result.adjacency_map)
        return session

    def _build_segments(
        self,
        groups: list[EntityGroup],
        adjacency: dict[int, set[int]],
    ) -> list[ReviewSegment]:
        nodes = {group.group_id for group in groups if not is_size_boundary(group)}
        components = _components_for_nodes(adjacency, nodes)
        segments: list[ReviewSegment] = []
        for segment_id, group_ids in enumerate(components):
            visible = [groups[group_id] for group_id in group_ids if _reviewable(groups[group_id])]
            if not visible:
                continue
            size, specification = _segment_defaults(visible)
            counts = Counter(group.name for group in visible)
            summary = "、".join(
                f"{name}×{count}" for name, count in sorted(counts.items(), key=lambda item: item[0])
            )
            mismatches = _segment_mismatches(visible)
            segments.append(ReviewSegment(
                segment_id=segment_id,
                group_ids=group_ids,
                size=size,
                specification=specification,
                component_count=len(visible),
                mismatch_count=sum(len(issues) for issues in mismatches.values()),
                component_summary=summary,
                color=SEGMENT_PALETTE[segment_id % len(SEGMENT_PALETTE)],
                mismatches=mismatches,
            ))
        segments.sort(key=lambda segment: (-segment.mismatch_count, segment.component_count, segment.segment_id))
        for order, segment in enumerate(segments, 1):
            segment.order = order
        return segments

    def to_frame(self, session: ReviewSession) -> pd.DataFrame:
        """建立 Gradio 可編輯表格；只有尺寸與規格欄位會被套用。"""
        return pd.DataFrame([
            {
                "優先順序": segment.order,
                "連通塊": segment.code,
                "風險標記": segment.risk_label,
                "尺寸": segment.size,
                "規格": segment.specification,
                "元件摘要": segment.component_summary,
            }
            for segment in session.segments
        ])

    def apply(self, session: ReviewSession, edited: pd.DataFrame) -> BomAnalysisResult:
        """套用人工尺寸／規格，保護固定元件規則，重算 Reducer 與最終 BOM。"""
        if edited is None or edited.empty:
            raise ValueError("連通塊覆核表不可為空")
        edited_by_code = {
            str(row["連通塊"]).strip(): row
            for _, row in edited.fillna("").iterrows()
            if str(row.get("連通塊", "")).strip()
        }
        segment_by_node: dict[int, ReviewSegment] = {}
        for segment in session.segments:
            row = edited_by_code.get(segment.code)
            if row is not None:
                segment.size = _normalize_size(str(row.get("尺寸", "")).strip())
                segment.specification = re.sub(r"\s+", " ", str(row.get("規格", "")).strip()).upper()
            for group_id in segment.group_ids:
                segment_by_node[group_id] = segment
                self._apply_segment_values(
                    session.result.groups[group_id], segment, session.result.groups, session.result.adjacency_map
                )

        self._rebuild_reducers(session.result.groups, session.result.adjacency_map, segment_by_node)
        bom, details = self.report_builder.build(session.result.groups)
        details.attrs.update(session.result.details.attrs)
        session.result.bom = bom
        session.result.details = details
        session.segments = self._build_segments(
            session.result.groups, session.result.adjacency_map
        )
        return session.result

    @staticmethod
    def _apply_segment_values(
        group: EntityGroup,
        segment: ReviewSegment,
        groups: list[EntityGroup],
        adjacency: dict[int, set[int]],
    ) -> None:
        if not _reviewable(group):
            return
        canonical = canonical_name(group.name)
        if has_fixed_quarter_inch_size(group, groups, adjacency):
            group.size = '1/4"'
        elif not is_hose(group):
            group.size = segment.size

        if is_tube(group) or is_size_boundary(group) or canonical in FINISH_MATERIAL_NAMES:
            group.material = segment.specification

    @staticmethod
    def _rebuild_reducers(
        groups: list[EntityGroup],
        adjacency: dict[int, set[int]],
        segment_by_node: dict[int, ReviewSegment],
    ) -> None:
        for group in groups:
            if not is_size_boundary(group):
                continue
            neighbor_sizes = []
            for neighbor in sorted(adjacency[group.group_id]):
                if neighbor not in segment_by_node:
                    continue
                neighbor_group = groups[neighbor]
                size = (
                    '1/4"'
                    if has_fixed_quarter_inch_size(neighbor_group, groups, adjacency)
                    else segment_by_node[neighbor].size
                )
                if size:
                    neighbor_sizes.append(size)
            group.size = canonical_reducer_size(neighbor_sizes)
            if not group.size:
                found = "、".join(dict.fromkeys(size for size in neighbor_sizes if size)) or "無"
                raise ValueError(
                    f"G{group.group_id}:{group.name} 為 Reducer類，但相鄰方向"
                    f"不是剛好兩種尺寸（找到：{found}）"
                )
