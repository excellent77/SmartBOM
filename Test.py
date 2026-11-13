"""SmartBOM 氣體文字材質法＋連通塊尺寸投票法的實驗版。

幾何載入、同色元件分組、重複線段移除與鄰接矩陣建立沿用現有
``smartbom_bom`` 實作；本檔案獨立負責文字分類、氣體材質、尺寸投票、
Tube 類型／長度與最終 BOM。本實驗版不會覆寫現行 ``*_bom.csv``，而是輸出
``*_test_bom.csv``、``*_test_details.csv`` 與 ``*_test_adjacency.csv``。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from smartbom_bom.config import BomSettings, TUBE_COLOR
from smartbom_bom.grouping import EntityGroupingService, _components_for_nodes, _connected_components
from smartbom_bom.io import DefaultVectorDataLoader
from smartbom_bom.models import BomAnalysisResult, EntityGroup
from smartbom_bom.predicates import (
    canonical_name, is_gauge, is_hose, is_size_boundary, is_special_rtee,
)
from smartbom_bom.review import ConnectivityReviewService
from smartbom_bom.text import (
    HOSE_RE, REDUCER_RE, SIZE_TOKEN, _best_text_match, _extract_size,
    _normalize_size, clean_dxf_text, extract_texts, majority_first,
)
from smartbom_bom.visualization import build_debug_figure, figure_to_html


# ===== 輸入、輸出與可調整參數 =====
SUPPORTED_EXTENSIONS = {".dxf", ".dwg", ".csv"}
TEST_BOM_SUFFIX = "_test_bom.csv"
TEST_DETAILS_SUFFIX = "_test_details.csv"
TEST_ADJACENCY_SUFFIX = "_test_adjacency.csv"
LENGTH_COMPONENT_NAMES = {"tube", "coil tube", "over tube", "hose", "軟管"}
SURFACE_FINISH_COMPONENT_NAMES = {
    "tee", "r.tee", "r tee", "reducer", "reducer-union", "reducer union",
    "elbow", "s-gland", "s gland", "l-gland", "l gland", "l_gland",
}


gas_material_dict = {
    "CDA": {"材質": "SUS304L", "表面處理方式": "AP"},
    "PCDA": {"材質": "SUS316L", "表面處理方式": "BA"},
    "HPN2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "XCDA": {"材質": "SUS316L", "表面處理方式": "EP"},
    "GN2": {"材質": "SUS316L", "表面處理方式": "BA"},
    "GO2": {"材質": "SUS316L", "表面處理方式": "BA"},
    "AN2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "AO2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "AHe": {"材質": "SUS316L", "表面處理方式": "EP"},
    "AAr": {"材質": "SUS316L", "表面處理方式": "EP"},
    "BA": {"材質": "SUS304L", "表面處理方式": "AP"},
    "NG": {"材質": "SUS304L", "表面處理方式": "AP"},
    "VENT": {"材質": "SUS316L", "表面處理方式": "EP"},
    "內連管": {"材質": "SUS316L", "表面處理方式": "EP"},
    "2.7%C2H4/He": {"材質": "SUS316L", "表面處理方式": "EP"},
    "10%CH4/Ar": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CO": {"材質": "SUS316L", "表面處理方式": "EP"},
    "30%C2H4/He": {"材質": "SUS316L", "表面處理方式": "EP"},
    "C3H6": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CH2F2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "NH3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "NF3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "N2O": {"材質": "SUS316L", "表面處理方式": "EP"},
    "1%O2/Ar": {"材質": "SUS316L", "表面處理方式": "EP"},
    "30%O2/He": {"材質": "SUS316L", "表面處理方式": "EP"},
    "C5F8": {"材質": "SUS316L", "表面處理方式": "EP"},
    "F2/Kr/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "F2/Ar/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "1.25%Kr/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SF6": {"材質": "SUS316L", "表面處理方式": "EP"},
    "Ar/Xe/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CHF3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CO2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "C4F8": {"材質": "SUS316L", "表面處理方式": "EP"},
    "Xe": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CF4": {"材質": "SUS316L", "表面處理方式": "EP"},
    "4%H2/N2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "AH2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "10%GeH4/H2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "C2H2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "F2/Ar/Xe/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "5%B2H6/N2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CH3F": {"材質": "SUS316L", "表面處理方式": "EP"},
    "Kr": {"材質": "SUS316L", "表面處理方式": "EP"},
    "100%C2H4": {"材質": "SUS316L", "表面處理方式": "EP"},
    "CH4": {"材質": "SUS316L", "表面處理方式": "EP"},
    "C4F6": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SiH(CH3)3(TMS)": {"材質": "SUS316L", "表面處理方式": "EP"},
    "TSA": {"材質": "SUS316L", "表面處理方式": "EP"},
    "0.5%H2/He": {"材質": "SUS316L", "表面處理方式": "EP"},
    "Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "10%Xe/Ne": {"材質": "SUS316L", "表面處理方式": "EP"},
    "O3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "1%PH3/Ar": {"材質": "SUS316L", "表面處理方式": "EP"},
    "5%BCl3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "WF6": {"材質": "SUS316L", "表面處理方式": "EP"},
    "TEOS": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SiH4": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SiH2Cl2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "BF3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "1%PH3/He": {"材質": "SUS316L", "表面處理方式": "EP"},
    "BCl3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SO2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "Si2H6": {"材質": "SUS316L", "表面處理方式": "EP"},
    "HF": {"材質": "SUS316L", "表面處理方式": "EP"},
    "SiCL4": {"材質": "SUS316L", "表面處理方式": "EP"},
    "5%PH3/N2": {"材質": "SUS316L", "表面處理方式": "EP"},
    "ClF3": {"材質": "SUS316L", "表面處理方式": "EP"},
    "TiCL4": {"材質": "SUS316L", "表面處理方式": "EP"},
}


TUBE_DESCRIPTION_RE = re.compile(
    rf"^\s*(?P<size>{SIZE_TOKEN})\s+(?P<material>\S+)\s+"
    rf"(?P<finish>[A-Za-z]{{2}})(?P<extra>.*?)\s*-\s*"
    rf"(?P<length>\d+(?:\.\d+)?)\s*M\s*$",
    re.IGNORECASE,
)
COLOR_NAME_RE = re.compile(r"顏色_\d+$")


class SpecificationError(ValueError):
    """文字規格無法唯一決定材質或尺寸時中止分析。"""


@dataclass(slots=True)
class TextClassification:
    gas_texts: list[dict]
    size_texts: list[dict]
    tube_descriptions: list[dict]


def _is_reviewable(group: EntityGroup) -> bool:
    return group.name != "IGNORE" and not COLOR_NAME_RE.fullmatch(str(group.name).strip())


def _gas_name(text: str) -> str | None:
    """用最長前綴找出氣體名稱，容許後方另有尺寸（如 CDA 15A）。"""
    cleaned = clean_dxf_text(text).strip()
    folded = cleaned.casefold()
    for name in sorted(gas_material_dict, key=len, reverse=True):
        key = name.casefold()
        if folded == key or (folded.startswith(key) and folded[len(key):len(key) + 1].isspace()):
            return name
    return None


def _parse_tube_description(text: str) -> dict | None:
    match = TUBE_DESCRIPTION_RE.fullmatch(clean_dxf_text(text))
    if not match:
        return None
    extra = re.sub(r"\s+", " ", match.group("extra")).strip()
    if re.search(r"coil\s*tube", extra, re.IGNORECASE):
        name = "Coil Tube"
    elif re.search(r"over\s*tube", extra, re.IGNORECASE):
        name = "Over Tube"
    else:
        name = "Tube"
    finish = match.group("finish").upper()
    return {
        "name": name,
        "size": _normalize_size(match.group("size")),
        "material": f'{match.group("material")} {finish}'.strip(),
        "base_material": match.group("material"),
        "finish": finish,
        "length": float(match.group("length")),
    }


def classify_texts(texts: Sequence[dict]) -> TextClassification:
    """已知數字開頭氣體優先於「數字開頭即尺寸」規則。"""
    gas_texts, size_texts, tube_descriptions, unknown = [], [], [], []
    for text in texts:
        value = clean_dxf_text(text["text"])
        gas = _gas_name(value)
        if gas:
            gas_texts.append({**text, "gas": gas})
            continue
        if re.match(r"^\s*\d", value):
            tube = _parse_tube_description(value)
            if tube:
                tube_descriptions.append({**text, **tube})
            size = _extract_size(value)
            if size:
                size_texts.append({**text, "size": size})
            continue
        unknown.append(f'第{text["index"]}列 {value!r}')
    if unknown:
        preview = "\n".join(unknown[:20])
        more = f"\n…其餘 {len(unknown) - 20} 筆" if len(unknown) > 20 else ""
        raise SpecificationError(
            "以下非數字開頭文字找不到 gas_material_dict 對應：\n"
            f"{preview}{more}"
        )
    return TextClassification(gas_texts, size_texts, tube_descriptions)


def _nearest_group(text: dict, groups: Sequence[EntityGroup]) -> EntityGroup:
    candidates = [
        group for group in groups
        if group.color != TUBE_COLOR and _is_reviewable(group)
    ]
    if not candidates:
        raise SpecificationError(f'氣體文字 {text["text"]!r} 附近沒有可用元件')
    return min(
        candidates,
        key=lambda group: (math.dist(text["position"], group.center), group.group_id),
    )


def _needs_surface_finish(group: EntityGroup) -> bool:
    return canonical_name(group.name) in SURFACE_FINISH_COMPONENT_NAMES


def apply_gas_materials(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    gas_texts: Sequence[dict],
) -> dict[int, dict]:
    """將氣體文字指定給最近非 Tube 元件，再傳遞至整個完整連通塊。"""
    components = _connected_components(adjacency)
    component_by_node = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    specification_by_component: dict[int, dict] = {}
    for text in sorted(gas_texts, key=lambda item: item["index"]):
        nearest = _nearest_group(text, groups)
        component_index = component_by_node[nearest.group_id]
        gas = text["gas"]
        specification = gas_material_dict[gas]
        previous = specification_by_component.get(component_index)
        if previous and (
            previous["材質"] != specification["材質"]
            or previous["表面處理方式"] != specification["表面處理方式"]
        ):
            raise SpecificationError(
                f"連通塊 {component_index} 同時配到衝突氣體 "
                f'{previous["gas"]!r} 與 {gas!r}'
            )
        specification_by_component[component_index] = {
            **specification, "gas": gas, "text": text["text"],
        }

    missing = []
    for component_index, component in enumerate(components):
        # 純 Tube 連通塊的材質由 Tube 規格文字決定，不強制要求氣體文字。
        relevant = [
            groups[node] for node in component
            if groups[node].color != TUBE_COLOR and _is_reviewable(groups[node])
        ]
        if relevant and component_index not in specification_by_component:
            missing.append(
                f'{component_index}({", ".join(f"G{group.group_id}:{group.name}" for group in relevant[:8])})'
            )
    if missing:
        raise SpecificationError("以下連通塊找不到氣體說明文字：" + "、".join(missing))

    node_specification = {}
    for component_index, component in enumerate(components):
        specification = specification_by_component.get(component_index)
        if not specification:
            continue
        base = specification["材質"]
        finish = specification["表面處理方式"]
        for node in component:
            group = groups[node]
            node_specification[node] = specification
            if not _is_reviewable(group):
                continue
            group.material = f"{base} {finish}" if _needs_surface_finish(group) else base
            group.notes.append(f'氣體 {specification["gas"]}: {specification["text"]}')
    return node_specification


def assign_tube_descriptions(
    groups: Sequence[EntityGroup], descriptions: Sequence[dict],
) -> dict[int, dict]:
    """以全域距離由近至遠一對一配對 Tube 文字與 Tube 圖元。"""
    tubes = [group for group in groups if group.color == TUBE_COLOR]
    candidates = sorted(
        (
            math.dist(text["position"], tube.center), text["index"], tube.group_id,
            text,
        )
        for text in descriptions for tube in tubes
    )
    assignments: dict[int, dict] = {}
    used_texts: set[int] = set()
    for distance, text_index, group_id, text in candidates:
        if text_index in used_texts or group_id in assignments:
            continue
        assignments[group_id] = {**text, "distance": distance}
        used_texts.add(text_index)
    unmatched = [text["text"] for text in descriptions if text["index"] not in used_texts]
    if unmatched:
        raise SpecificationError("Tube 規格文字無法配對唯一 Tube：" + "、".join(unmatched))
    return assignments


def _component_text_matches(
    groups: Sequence[EntityGroup], size_texts: Sequence[dict],
) -> dict[int, dict]:
    """一般元件只找最近的數字開頭尺寸文字。"""
    matches = {}
    for group in groups:
        if group.color == TUBE_COLOR or is_size_boundary(group) or not _is_reviewable(group):
            continue
        best = _best_text_match(group.center, size_texts)
        if best:
            distance, text, _ = best
            matches[group.group_id] = {**text, "distance": distance}
            group.matched_text = text["text"]
    return matches


def _canonical_reducer_size(values: Iterable[str]) -> str:
    unique = list(dict.fromkeys(_normalize_size(value) for value in values if value))
    return "x".join(sorted(unique, reverse=True)[:2])


def apply_sizes_and_lengths(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    classification: TextClassification,
) -> tuple[dict[int, str], dict[int, dict], dict[int, dict]]:
    """移除 Reducer 後進行尺寸投票，再套用 Gauge、R.Tee、Hose 規則。"""
    component_matches = _component_text_matches(groups, classification.size_texts)
    tube_assignments = assign_tube_descriptions(groups, classification.tube_descriptions)
    nodes = {group.group_id for group in groups if not is_size_boundary(group)}
    segments = _components_for_nodes(adjacency, nodes)
    uniform_size_by_node: dict[int, str] = {}

    for segment in segments:
        votes = sorted(
            (
                component_matches[node]["index"],
                component_matches[node]["size"],
            )
            for node in segment if node in component_matches
        )
        # 非 Tube 元件沒有尺寸證據時，才使用該段 Tube 規格文字作為備援票。
        if not votes:
            votes = sorted(
                (tube_assignments[node]["index"], tube_assignments[node]["size"])
                for node in segment if node in tube_assignments
            )
        uniform_size = majority_first([size for _, size in votes])
        if not uniform_size:
            names = ", ".join(f"G{node}:{groups[node].name}" for node in segment[:10])
            raise SpecificationError(f"連通尺寸段找不到任何尺寸票：{names}")
        for node in segment:
            uniform_size_by_node[node] = uniform_size

    for group in groups:
        if is_size_boundary(group):
            continue
        uniform_size = uniform_size_by_node.get(group.group_id, "")
        description = tube_assignments.get(group.group_id)
        if group.color == TUBE_COLOR:
            if description:
                group.name = description["name"]
                group.size = description["size"]
                group.material = description["material"]
                group.quantity = description["length"]
                group.matched_text = description["text"]
            else:
                group.size = uniform_size
                group.quantity = 0.0
            continue
        if is_gauge(group):
            group.size = '1/4"'
        elif canonical_name(group.name) == "gasket" and any(
            is_gauge(groups[neighbor]) for neighbor in adjacency[group.group_id]
        ):
            group.size = '1/4"'
        elif is_special_rtee(group):
            group.size = f'{uniform_size}x1/4"'
        elif is_hose(group):
            nearest = _best_text_match(group.center, classification.size_texts, HOSE_RE)
            if nearest:
                _, text, match = nearest
                group.size = _normalize_size(match.group("size"))
                group.quantity = float(match.group("length"))
                group.matched_text = text["text"]
            else:
                group.size = uniform_size
                group.quantity = 0.0
                group.notes.append("找不到軟管長度描述")
        else:
            group.size = uniform_size
            group.quantity = 1.0

    for group in groups:
        if not is_size_boundary(group):
            continue
        neighbor_sizes = [
            uniform_size_by_node.get(neighbor, groups[neighbor].size)
            for neighbor in sorted(adjacency[group.group_id])
        ]
        group.size = _canonical_reducer_size(neighbor_sizes)
        if not group.size:
            nearest = _best_text_match(group.center, classification.size_texts, REDUCER_RE)
            if nearest:
                _, text, match = nearest
                group.size = _canonical_reducer_size([
                    match.group("first"), match.group("second"),
                ])
                group.matched_text = text["text"]
            else:
                raise SpecificationError(f"G{group.group_id}:{group.name} 找不到兩側尺寸")
        group.quantity = 1.0
    return uniform_size_by_node, tube_assignments, component_matches


def _format_number(value: float) -> str:
    return str(int(value)) if math.isclose(value, round(value)) else f"{value:g}"


def build_reports(groups: Sequence[EntityGroup]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立包含 Over Tube 長度加總規則的 BOM 與群組明細。"""
    details = pd.DataFrame([{
        "群組編號": group.group_id, "品名": group.name, "顏色": group.color,
        "尺寸": group.size, "材質": group.material, "原始數量": group.quantity,
        "配對文字": group.matched_text, "中心 X": group.center[0], "中心 Y": group.center[1],
        "圖元數": len(group.geometries), "來源列號": ",".join(map(str, group.source_rows)),
        "備註": "；".join(group.notes),
    } for group in groups])
    aggregated: dict[tuple[str, str, str], float] = {}
    for group in groups:
        if not _is_reviewable(group):
            continue
        key = (group.name, group.size, group.material)
        is_length = canonical_name(group.name) in LENGTH_COMPONENT_NAMES
        if is_length and group.quantity <= 0:
            continue
        aggregated[key] = aggregated.get(key, 0.0) + (group.quantity if is_length else 1.0)
    bom = pd.DataFrame([
        {
            "品名": name, "尺寸": size, "材質": material,
            "數量": f"{_format_number(quantity)}M"
            if canonical_name(name) in LENGTH_COMPONENT_NAMES else int(quantity),
        }
        for (name, size, material), quantity in aggregated.items()
    ], columns=["品名", "尺寸", "材質", "數量"])
    return bom, details


class TestBomAnalyzer:
    """實驗版分析流程的單一協調者。"""

    def __init__(self, settings: BomSettings | None = None):
        self.settings = settings or BomSettings()
        self.loader = DefaultVectorDataLoader()
        self.grouping = EntityGroupingService(
            self.settings.tube_color, self.settings.color_component_map,
        )

    def analyze(self, vectors: pd.DataFrame) -> BomAnalysisResult:
        self.settings.validate()
        groups, removed_rows = self.grouping.build_groups(
            vectors, self.settings.component_tolerance, self.settings.duplicate_tolerance,
        )
        matrix, adjacency = self.grouping.build_adjacency(
            groups, self.settings.connection_tolerance,
        )
        texts = extract_texts(vectors)
        classification = classify_texts(texts)
        apply_gas_materials(groups, adjacency, classification.gas_texts)
        apply_sizes_and_lengths(groups, adjacency, classification)
        bom, details = build_reports(groups)
        details.attrs["removed_duplicate_rows"] = removed_rows
        labels = [f"G{group.group_id}:{group.name}" for group in groups]
        adjacency_frame = pd.DataFrame(matrix, index=labels, columns=labels)
        adjacency_frame.index.name = "群組"
        return BomAnalysisResult(
            bom=bom, details=details, adjacency=adjacency_frame,
            groups=list(groups), adjacency_map=adjacency, texts=texts,
        )

    def run(self, file_path: str | Path, write_reports: bool = True) -> BomAnalysisResult:
        source = Path(file_path)
        if not source.is_file():
            raise FileNotFoundError(f"找不到輸入檔案：{source}")
        result = self.analyze(self.loader.load(source))
        if write_reports:
            write_test_reports(source, result)
        return result


def output_paths(source: str | Path) -> tuple[Path, Path, Path]:
    path = Path(source)
    return (
        path.with_name(f"{path.stem}{TEST_BOM_SUFFIX}"),
        path.with_name(f"{path.stem}{TEST_DETAILS_SUFFIX}"),
        path.with_name(f"{path.stem}{TEST_ADJACENCY_SUFFIX}"),
    )


def write_test_reports(source: str | Path, result: BomAnalysisResult) -> tuple[Path, Path, Path]:
    paths = output_paths(source)
    result.bom.to_csv(paths[0], index=False, encoding="utf-8-sig")
    result.details.to_csv(paths[1], index=False, encoding="utf-8-sig")
    result.adjacency.to_csv(paths[2], encoding="utf-8-sig")
    return paths


# ===== Gradio 實驗介面 =====
EMPTY_BOM = pd.DataFrame(columns=["品名", "尺寸", "材質", "數量"])
EMPTY_FRAME = pd.DataFrame()


def analyze_uploaded_file(file_path: str | Path | None):
    if not file_path:
        return "請先上傳 DXF、DWG 或向量 CSV。", "", EMPTY_BOM, EMPTY_FRAME, EMPTY_FRAME, []
    source = Path(str(file_path))
    if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        return f"❌ 不支援的檔案格式：{source.suffix}", "", EMPTY_BOM, EMPTY_FRAME, EMPTY_FRAME, []
    try:
        analyzer = TestBomAnalyzer()
        vectors = analyzer.loader.load(source)
        result = analyzer.analyze(vectors)
        paths = write_test_reports(source, result)
        review = ConnectivityReviewService()
        session = review.create_session(source, vectors, result)
        figure = figure_to_html(
            build_debug_figure(vectors, result.details, result.bom, session.segments)
        )
        status = (
            f"✅ **實驗版分析完成：{source.name}**  \n"
            f"- BOM：{len(result.bom)} 筆  \n"
            f"- 圖元群組：{len(result.groups)} 組  \n"
            f"- 文字：{len(result.texts)} 筆"
        )
        return status, figure, result.bom, result.details, result.adjacency.reset_index(), [str(path) for path in paths]
    except Exception as exc:
        return f"❌ **分析停止**：{type(exc).__name__}: {exc}", "", EMPTY_BOM, EMPTY_FRAME, EMPTY_FRAME, []


def create_interface() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("尚未安裝 Gradio，請先安裝 requirements.txt。") from exc
    with gr.Blocks(title="SmartBOM 規格配對實驗版") as demo:
        gr.Markdown(
            "# SmartBOM 規格配對實驗版\n"
            "材質由氣體文字與 gas_material_dict 決定；尺寸由移除 Reducer 後的連通段投票決定。"
        )
        with gr.Row():
            upload = gr.File(
                label="上傳 DXF、DWG 或向量 CSV",
                file_types=[".dxf", ".dwg", ".csv"], type="filepath",
            )
            status = gr.Markdown("請上傳檔案開始分析。")
        with gr.Tabs():
            with gr.Tab("① 互動圖"):
                plot = gr.HTML()
            with gr.Tab("② BOM"):
                bom = gr.Dataframe(value=EMPTY_BOM, interactive=False, wrap=True)
            with gr.Tab("③ 群組明細"):
                details = gr.Dataframe(value=EMPTY_FRAME, interactive=False, wrap=True)
            with gr.Tab("④ 鄰接矩陣"):
                adjacency = gr.Dataframe(value=EMPTY_FRAME, interactive=False, wrap=False)
        downloads = gr.File(label="下載實驗版報表", file_count="multiple")
        upload.upload(
            analyze_uploaded_file, upload,
            [status, plot, bom, details, adjacency, downloads],
            show_progress="full",
        )
    return demo


def main(file_path: str | Path | None = None) -> None:
    """傳入檔案路徑時直接分析；未傳入時啟動 Gradio 介面。"""
    if file_path is not None:
        result = TestBomAnalyzer().run(file_path, write_reports=True)
        print(result.bom.to_string(index=False))
        return
    create_interface().queue().launch()


if __name__ == "__main__":
    main()
