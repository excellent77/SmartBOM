"""以去重、圖元組鄰接關係及連通塊投票產生工程 BOM。

輸入 DXF/DWG 時會直接呼叫 dxf_reader.read_dwg()。顏色 30 視為 Tube，
每條管線圖元各自成組；其餘非文字圖元依顏色及幾何連通性聚類成元件。
"""

import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from dxf_reader import read_dwg
# ===== 可調整參數 =====
COMPONENT_TOLERANCE = 0.5
CONNECTION_TOLERANCE = 0.5
DUPLICATE_TOLERANCE = 0.01
SEGMENT_INTERSECTION_EPSILON = 1e-9
CIRCLE_SAMPLE_STEPS = 72
ARC_SAMPLE_STEPS = 36
MIN_ARC_SAMPLE_STEPS = 8
DISTANCE_SCORE_WEIGHT = 100.0
HORIZONTAL_ALIGNMENT_BONUS = 5.0
VERTICAL_ALIGNMENT_BONUS = 5.0
ALIGNMENT_ABSOLUTE_TOLERANCE = 1.0
ALIGNMENT_SLOPE_TOLERANCE = 0.15
TUBE_COLOR = "顏色_30"
BOM_OUTPUT_SUFFIX = "_connectivity_vote_bom.csv"
DETAILS_OUTPUT_SUFFIX = "_connectivity_vote_bom_details.csv"
ADJACENCY_OUTPUT_SUFFIX = "_connectivity_vote_bom_adjacency.csv"

from columns import (
    COL_NAME, COL_COLOR, COL_COUNT,
    COL_START_X, COL_START_Y, COL_END_X, COL_END_Y,
    COL_CENTER_X, COL_CENTER_Y,
    COL_POS_X, COL_POS_Y, COL_VALUE, COL_ROTATION, COL_RADIUS,
    ETYPE_LINE, ETYPE_POLYLINE, ETYPE_CIRCLE, ETYPE_ELLIPSE,
    ETYPE_ARC, ETYPE_TEXT, ETYPE_MTEXT, ETYPE_HATCH,
)

# 依 GAS零件圖塊對照表_v5 更新；此映射只屬於本程式。
COLOR_COMPONENT_MAP = {
    "顏色_181": {"name": "Ball-valve"},
    "顏色_3": {"name": "Dia-valve"},
    "顏色_24": {"name": "Bellow valve"},
    "顏色_212": {"name": "C.V"},
    "顏色_140": {"name": "3P Regulator"},
    "顏色_5": {"name": "Reducer"},
    "顏色_16": {"name": "ELBOW"},
    "顏色_1": {"name": "Nut(F)"},
    "顏色_8": {"name": "NUT(M)"},
    "顏色_142": {"name": "S-Gland"},
    "顏色_11": {"name": "L-Gland"},
    "顏色_165": {"name": "Gasket"},
    "顏色_171": {"name": "Union R.Tee"},
    "顏色_241": {"name": "Union Tee"},
    "顏色_67": {"name": "Union"},
    "顏色_211": {"name": "Reducer-Union"},
    "顏色_4": {"name": "軟管"},
    "顏色_37": {"name": "Cap"},
    "顏色_6": {"name": "Regulator(SWG)"},
    "顏色_122": {"name": "Regulator(FVCR)"},
    "顏色_33": {"name": "Tee"},
    "顏色_104": {"name": "R.Tee"},
    "顏色_153": {"name": "VCR Tee"},
    "顏色_225": {"name": "VCR R.Tee"},
    "顏色_40": {"name": "SWG Gauge(背接式)"},
    "顏色_192": {"name": "VCR Gauge(背接式)"},
    "顏色_252": {"name": "IGNORE"},  # 表上註記「不須列入計算的材料」，應在後製時直接排除，不當成 BOM 品項
}

TEXT_TYPES = {ETYPE_TEXT, ETYPE_MTEXT, "TEXT", "MTEXT"}
GEOMETRY_TYPES = {ETYPE_LINE, ETYPE_POLYLINE, ETYPE_CIRCLE, ETYPE_ARC, ETYPE_ELLIPSE, ETYPE_HATCH, "LINE", "CIRCLE", "ARC", "ELLIPSE"}


@dataclass
class Geometry:
    row_index: int
    color: str
    kind: str
    points: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.points[:, 0].min()), float(self.points[:, 1].min()),
            float(self.points[:, 0].max()), float(self.points[:, 1].max()),
        )


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def read_vector_csv(path: str | Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("無法判斷 CSV 編碼：" + "; ".join(errors))


def load_vector_data(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    extension = input_path.suffix.casefold()
    if extension in {".dxf", ".dwg"}:
        return read_dwg(str(input_path))
    if extension == ".csv":
        return read_vector_csv(input_path)
    raise ValueError(f"不支援的輸入格式：{extension}；請使用 .dxf、.dwg 或 .csv")


def _number(row: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _point(row: pd.Series, x_columns: Sequence[str], y_columns: Sequence[str]) -> tuple[float, float] | None:
    x, y = _number(row, *x_columns), _number(row, *y_columns)
    return None if x is None or y is None else (x, y)


def _sample_circle(
    center: tuple[float, float],
    radius: float,
    steps: int = CIRCLE_SAMPLE_STEPS,
) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, steps + 1)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def _sample_arc(row: pd.Series, center: tuple[float, float], radius: float) -> np.ndarray:
    start = _point(row, (COL_START_X,), (COL_START_Y,))
    end = _point(row, (COL_END_X,), (COL_END_Y,))
    if start and end:
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    else:
        start_deg, total_deg = _number(row, "起始角度", "角度1"), _number(row, "總角度")
        if start_deg is None or total_deg is None:
            return _sample_circle(center, radius)
        start_angle = math.radians(start_deg)
        end_angle = start_angle + math.radians(total_deg)
    while end_angle < start_angle:
        end_angle += 2 * math.pi
    count = max(
        MIN_ARC_SAMPLE_STEPS,
        int(ARC_SAMPLE_STEPS * (end_angle - start_angle) / (2 * math.pi)),
    )
    angles = np.linspace(start_angle, end_angle, count + 1)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def row_to_geometry(index: int, row: pd.Series) -> Geometry | None:
    kind, color = str(row.get(COL_NAME, "")).strip(), str(row.get(COL_COLOR, "")).strip()
    if kind not in GEOMETRY_TYPES or not color:
        return None
    start = _point(row, (COL_START_X,), (COL_START_Y,))
    end = _point(row, (COL_END_X,), (COL_END_Y,))
    center = _point(row, (COL_CENTER_X,), (COL_CENTER_Y,))
    radius = _number(row, COL_RADIUS)
    if kind in {ETYPE_LINE, ETYPE_POLYLINE, "LINE"} and start and end:
        points = np.asarray([start, end], dtype=float)
    elif kind in {ETYPE_CIRCLE, "CIRCLE"} and center and radius is not None:
        points = _sample_circle(center, abs(radius))
    elif kind in {ETYPE_ARC, "ARC"} and center and radius is not None:
        points = _sample_arc(row, center, abs(radius))
    elif kind == ETYPE_HATCH and start and end:
        x1, y1 = start
        x2, y2 = end
        points = np.asarray([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dtype=float)
    elif kind in {ETYPE_ELLIPSE, "ELLIPSE"} and center:
        points = np.asarray([center], dtype=float)
    else:
        return None
    return Geometry(index, color, kind, points)


def clean_dxf_text(value: str) -> str:
    text = str(value).replace("\\P", " ").replace("\\~", " ").replace("^I", " ")
    text = re.sub(r"\\[AaCcFfHhQqTtWw][^;]*;", "", text)
    text = re.sub(r"\\[LlOoKk]", "", text).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _text_center(row: pd.Series, text: str, insert: tuple[float, float]) -> tuple[float, float]:
    """回傳 DXF 提供的文字位置，不再以字數及字高估算文字中心。"""
    return insert


def extract_texts(df: pd.DataFrame) -> list[dict]:
    texts = []
    for index, row in df.iterrows():
        if str(row.get(COL_NAME, "")).strip() not in TEXT_TYPES:
            continue
        position = _point(row, (COL_POS_X, "位置 X1", "X"), (COL_POS_Y, "位置 Y1", "Y"))
        if position is None:
            continue
        value = ""
        for column in (COL_VALUE, "內容"):
            candidate = row.get(column)
            if pd.notna(candidate) and str(candidate).strip():
                value = clean_dxf_text(candidate)
                break
        if value:
            visual_center = _point(row, ("文字中心 X",), ("文字中心 Y",))
            texts.append({
                "index": int(index),
                "text": value,
                "position": visual_center or _text_center(row, value, position),
            })
    return texts


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 == 0:
        return float(np.linalg.norm(point - start))
    ratio = min(1.0, max(0.0, float(np.dot(point - start, delta) / length2)))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    epsilon: float = SEGMENT_INTERSECTION_EPSILON,
) -> bool:
    ab, cd = b - a, d - c
    values = (_cross(ab, c - a), _cross(ab, d - a), _cross(cd, a - c), _cross(cd, b - c))
    proper = ((values[0] > epsilon and values[1] < -epsilon) or (values[0] < -epsilon and values[1] > epsilon)) and (
        (values[2] > epsilon and values[3] < -epsilon) or (values[2] < -epsilon and values[3] > epsilon)
    )
    if proper:
        return True
    return min(
        _point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b), _point_segment_distance(d, a, b),
    ) <= epsilon


def _segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b), _point_segment_distance(d, a, b),
    )


def geometry_distance(first: Geometry, second: Geometry, stop_at: float = math.inf) -> float:
    left, right = first.points, second.points
    left_segments = list(zip(left[:-1], left[1:])) if len(left) > 1 else [(left[0], left[0])]
    right_segments = list(zip(right[:-1], right[1:])) if len(right) > 1 else [(right[0], right[0])]
    best = math.inf
    for a, b in left_segments:
        for c, d in right_segments:
            best = min(best, _segment_distance(a, b, c, d))
            if best <= stop_at:
                return best
    return best


def cluster_geometries(geometries: Sequence[Geometry], tolerance: float) -> list[list[Geometry]]:
    groups, by_color = [], {}
    for geometry in geometries:
        by_color.setdefault(geometry.color, []).append(geometry)
    for items in by_color.values():
        dsu = DisjointSet(len(items))
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                a, b = items[left].bbox, items[right].bbox
                bbox_near = not (
                    a[2] + tolerance < b[0] or b[2] + tolerance < a[0]
                    or a[3] + tolerance < b[1] or b[3] + tolerance < a[1]
                )
                if bbox_near and geometry_distance(items[left], items[right], tolerance) <= tolerance:
                    dsu.union(left, right)
        connected = {}
        for index, geometry in enumerate(items):
            connected.setdefault(dsu.find(index), []).append(geometry)
        groups.extend(connected.values())
    return groups


def component_center(component: Sequence[Geometry]) -> tuple[float, float]:
    points = np.vstack([geometry.points for geometry in component])
    return float((points[:, 0].min() + points[:, 0].max()) / 2), float((points[:, 1].min() + points[:, 1].max()) / 2)


FINISH_MATERIAL_NAMES = {
    "tee", "r.tee", "r tee", "reducer", "elbow", "s gland",
    "s-gland", "l gland", "l-gland", "cap",
}

SIZE_TOKEN = r'(?:\d+(?:/\d+)?"|\d+(?:\.\d+)?A|\d+(?:\.\d+)?\s*mm)'
TUBE_RE = re.compile(
    rf"^\s*(?P<size>{SIZE_TOKEN})\s+(?P<material>\S+)\s+"
    rf"(?P<finish>[A-Za-z]{{2}})(?P<extra>.*?)\s*-\s*"
    rf"(?P<length>\d+(?:\.\d+)?)\s*M\s*$",
    re.IGNORECASE,
)
HOSE_RE = re.compile(
    rf"^\s*(?P<size>{SIZE_TOKEN})\s+軟管\s*-\s*(?P<length>\d+(?:\.\d+)?)\s*M\s*$",
    re.IGNORECASE,
)
REDUCER_RE = re.compile(
    rf"(?P<first>{SIZE_TOKEN})\s*[xX×]\s*(?P<second>{SIZE_TOKEN})(?:\s+R\.?\s*C\.?)?",
    re.IGNORECASE,
)


@dataclass
class EntityGroup:
    group_id: int
    name: str
    color: str
    geometries: list[Geometry]
    center: tuple[float, float]
    source_rows: list[int]
    size: str = ""
    material: str = ""
    quantity: float = 1.0
    matched_text: str = ""
    notes: list[str] = field(default_factory=list)


def _canonical_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().casefold())


def _format_number(value: float) -> str:
    return str(int(value)) if math.isclose(value, round(value)) else f"{value:g}"


def _text_match_score(
    component_position: tuple[float, float],
    text_position: tuple[float, float],
) -> tuple[float, float]:
    """綜合距離、水平對齊與垂直對齊計算文字配對分數。"""
    dx = abs(component_position[0] - text_position[0])
    dy = abs(component_position[1] - text_position[1])
    distance = math.hypot(dx, dy)
    score = DISTANCE_SCORE_WEIGHT / (1.0 + distance)
    if dy <= max(ALIGNMENT_ABSOLUTE_TOLERANCE, dx * ALIGNMENT_SLOPE_TOLERANCE):
        score += HORIZONTAL_ALIGNMENT_BONUS
    if dx <= max(ALIGNMENT_ABSOLUTE_TOLERANCE, dy * ALIGNMENT_SLOPE_TOLERANCE):
        score += VERTICAL_ALIGNMENT_BONUS
    return score, distance


def _best_text_match(point: tuple[float, float], texts: Sequence[dict], pattern: re.Pattern | None = None):
    candidates = []
    for text in texts:
        match = pattern.search(text["text"]) if pattern else None
        if pattern is not None and match is None:
            continue
        score, distance = _text_match_score(point, text["position"])
        candidates.append((score, distance, text, match))
    return max(candidates, key=lambda item: (item[0], -item[1], -item[2]["index"])) if candidates else None


def _normalize_size(value: str) -> str:
    value = re.sub(r"\s+", "", str(value).strip())
    if value.casefold().endswith("mm"):
        return f"{value[:-2]}mm"
    if value.casefold().endswith("a"):
        return f"{value[:-1]}A"
    return value


def _tube_description(text: str) -> dict | None:
    """解析 Tube/Coil Tube；加熱等附加描述不影響 BOM 規格。"""
    match = TUBE_RE.fullmatch(text)
    if not match:
        return None
    extra = re.sub(r"\s+", " ", match.group("extra")).strip()
    component_name = "Coil Tube" if re.search(r"coil\s*tube", extra, re.IGNORECASE) else "Tube"
    finish = match.group("finish").upper()
    return {
        "name": component_name,
        "size": _normalize_size(match.group("size")),
        "material": f'{match.group("material")} {finish}',
        "base_material": match.group("material"),
        "finish": finish,
        "length": float(match.group("length")),
    }


def _line_duplicate(first: Geometry, second: Geometry, tolerance: float) -> bool:
    """判斷同色兩線段是否高度重疊；允許起終點方向相反。"""
    if first.kind not in {"線", "聚合線", "LINE"} or second.kind not in {"線", "聚合線", "LINE"}:
        return False
    if len(first.points) != 2 or len(second.points) != 2:
        return False
    first_start, first_end = first.points
    second_start, second_end = second.points
    same_direction = max(
        float(np.linalg.norm(first_start - second_start)),
        float(np.linalg.norm(first_end - second_end)),
    ) <= tolerance
    reverse_direction = max(
        float(np.linalg.norm(first_start - second_end)),
        float(np.linalg.norm(first_end - second_start)),
    ) <= tolerance
    midpoint_near = float(np.linalg.norm(
        (first_start + first_end) / 2 - (second_start + second_end) / 2
    )) <= tolerance
    return midpoint_near and (same_direction or reverse_direction)


def remove_duplicate_lines(geometries: Sequence[Geometry], tolerance: float) -> tuple[list[Geometry], list[int]]:
    """只在相同顏色內去除重複線段，保留最先出現的原始列。"""
    kept: list[Geometry] = []
    removed_rows: list[int] = []
    lines_by_color: dict[str, list[Geometry]] = {}
    for geometry in geometries:
        if geometry.kind not in {"線", "聚合線", "LINE"}:
            kept.append(geometry)
            continue
        duplicates = lines_by_color.setdefault(geometry.color, [])
        if any(_line_duplicate(geometry, old, tolerance) for old in duplicates):
            removed_rows.append(geometry.row_index)
        else:
            duplicates.append(geometry)
            kept.append(geometry)
    return kept, removed_rows


def build_entity_groups(
    df: pd.DataFrame,
    tolerance: float,
    duplicate_tolerance: float,
) -> tuple[list[EntityGroup], list[int]]:
    geometries = [geometry for index, row in df.iterrows() if (geometry := row_to_geometry(int(index), row))]
    geometries, removed_rows = remove_duplicate_lines(geometries, duplicate_tolerance)
    tubes = [geometry for geometry in geometries if geometry.color == TUBE_COLOR]
    components = [geometry for geometry in geometries if geometry.color != TUBE_COLOR]

    raw_groups: list[tuple[str, str, list[Geometry]]] = []
    # 規格要求：每個顏色 30 圖元本身就是一組，不先彼此聚類。
    raw_groups.extend(("Tube", TUBE_COLOR, [geometry]) for geometry in tubes)
    for cluster in cluster_geometries(components, tolerance):
        color = cluster[0].color
        component_name = COLOR_COMPONENT_MAP.get(color, {}).get("name", color)
        raw_groups.append((component_name, color, cluster))

    groups = []
    for group_id, (name, color, items) in enumerate(raw_groups):
        groups.append(EntityGroup(
            group_id=group_id,
            name=name,
            color=color,
            geometries=items,
            center=component_center(items),
            source_rows=[item.row_index for item in items],
        ))
    return groups, removed_rows


def groups_connected(first: EntityGroup, second: EntityGroup, tolerance: float) -> bool:
    """任兩個圖元接觸、相交、重疊或足夠接近，即判定兩圖元組相連。"""
    for left in first.geometries:
        for right in second.geometries:
            a, b = left.bbox, right.bbox
            bbox_near = not (
                a[2] + tolerance < b[0] or b[2] + tolerance < a[0]
                or a[3] + tolerance < b[1] or b[3] + tolerance < a[1]
            )
            if bbox_near and geometry_distance(left, right, tolerance) <= tolerance:
                return True
    return False


def build_adjacency_matrix(groups: Sequence[EntityGroup], tolerance: float) -> tuple[np.ndarray, dict[int, set[int]]]:
    size = len(groups)
    matrix = np.zeros((size, size), dtype=np.uint8)
    adjacency = {index: set() for index in range(size)}
    for left in range(size):
        for right in range(left + 1, size):
            if groups_connected(groups[left], groups[right], tolerance):
                matrix[left, right] = matrix[right, left] = 1
                adjacency[left].add(right)
                adjacency[right].add(left)
    return matrix, adjacency


def assign_tube_descriptions(groups: Sequence[EntityGroup], texts: Sequence[dict]) -> dict[int, dict]:
    """以分輪競爭方式，將合法描述文字一對一配給 Tube。

    每輪所有未配對文字都選擇綜合分數最高的剩餘 Tube；若多筆文字競爭
    同一 Tube，僅分數最高者取得，其餘文字下一輪改從尚未占用 Tube 選擇。
    """
    available_tubes = {group.group_id for group in groups if group.color == TUBE_COLOR}
    assignments: dict[int, dict] = {}
    if not available_tubes:
        return assignments

    pending_texts = []
    for text in texts:
        description = _tube_description(text["text"])
        if description is not None:
            pending_texts.append((text, description))

    while pending_texts and available_tubes:
        proposals: dict[int, list[tuple[float, float, dict, dict]]] = {}
        for text, description in pending_texts:
            group_id = max(
                available_tubes,
                key=lambda node: (
                    _text_match_score(groups[node].center, text["position"])[0],
                    -_text_match_score(groups[node].center, text["position"])[1],
                    -node,
                ),
            )
            score, distance = _text_match_score(groups[group_id].center, text["position"])
            proposals.setdefault(group_id, []).append((score, distance, text, description))

        assigned_text_indices = set()
        for group_id, candidates in proposals.items():
            # 分數相同時依距離、文字列號固定結果，確保每次執行都一致。
            score, distance, text, description = max(
                candidates,
                key=lambda item: (item[0], -item[1], -item[2]["index"]),
            )
            assignments[group_id] = {
                **description,
                "text": text["text"],
                "text_index": text["index"],
                "score": score,
                "distance": distance,
            }
            available_tubes.remove(group_id)
            assigned_text_indices.add(text["index"])

        pending_texts = [
            (text, description)
            for text, description in pending_texts
            if text["index"] not in assigned_text_indices
        ]
    return assignments


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(adjacency)
    components = []
    while unseen:
        start = min(unseen)
        queue, found = deque([start]), []
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            found.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(found)
    return components


def _reducer_sizes(group: EntityGroup, texts: Sequence[dict]) -> tuple[str, str, str] | None:
    nearest = _best_text_match(group.center, texts, REDUCER_RE)
    if nearest is None:
        return None
    _, _, text, match = nearest
    return match.group("first"), match.group("second"), text["text"]


def _other_reducer_size(first: str, second: str, incoming: str) -> str:
    if incoming.casefold() == first.casefold():
        return second
    if incoming.casefold() == second.casefold():
        return first
    return second


def _apply_group_rules(group: EntityGroup, incoming_size: str, material: str, texts: Sequence[dict]) -> str:
    """設定目前群組 BOM 欄位，回傳 BFS 傳給下一層的尺寸。"""
    canonical = _canonical_name(group.name)
    group.size, group.material, group.quantity = incoming_size, material, 1.0

    if "gauge" in canonical or "guage" in canonical:
        group.size = '1/4"'
        return incoming_size  # Gauge 是支路元件，不改變主管尺寸。

    if canonical in {"union r.tee", "vcr r.tee"}:
        group.size = f'{incoming_size}x1/4"' if incoming_size else 'x1/4"'
        return incoming_size

    if canonical in {"hose", "軟管"}:
        nearest = _best_text_match(group.center, texts, HOSE_RE)
        if nearest:
            _, _, text, match = nearest
            group.size = match.group("size")
            group.quantity = float(match.group("length"))
            group.matched_text = text["text"]
        else:
            group.quantity = 0.0
            group.notes.append("找不到軟管長度描述")
        return group.size or incoming_size

    if canonical == "reducer":
        reducer = _reducer_sizes(group, texts)
        if reducer:
            first, second, text = reducer
            group.size = f"{first}x{second}"
            group.matched_text = text
            return _other_reducer_size(first, second, incoming_size)
        group.notes.append("找不到 Reducer 尺寸描述")
    return incoming_size


def fill_by_bfs(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
) -> dict[int, dict]:
    tube_descriptions = assign_tube_descriptions(groups, texts)

    for connected in _connected_components(adjacency):
        seeds = [node for node in connected if node in tube_descriptions]
        # 同一連通網若有多個合法起點，先使用文字離線段中點最近者。
        seed = min(seeds, key=lambda node: tube_descriptions[node]["distance"]) if seeds else min(connected)
        initial = tube_descriptions.get(seed, {"size": "", "material": "", "length": 0.0, "text": ""})
        queue = deque([(seed, "", initial["size"], initial["material"])])
        visited = set()

        while queue:
            node, parent, incoming_size, material = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            group = groups[node]

            if group.name == "Tube":
                description = tube_descriptions.get(node)
                group.size, group.material = incoming_size, material
                group.quantity = 0.0
                if description:
                    group.size = description["size"]
                    group.material = description["material"]
                    group.quantity = description["length"]
                    group.matched_text = description["text"]
                outgoing_size = group.size
                outgoing_material = group.material
            else:
                outgoing_size = _apply_group_rules(group, incoming_size, material, texts)
                outgoing_material = group.material

            for neighbor in sorted(adjacency[node]):
                if neighbor != parent and neighbor not in visited:
                    queue.append((neighbor, node, outgoing_size, outgoing_material))

        if not seeds:
            for node in connected:
                groups[node].notes.append("此連通區找不到合法 Tube 起點描述")
    return tube_descriptions


def majority_first(values: Sequence[str]) -> str:
    """回傳眾數；票數相同時保留輸入序列中先出現者。"""
    counts: dict[str, int] = {}
    first_position: dict[str, int] = {}
    for position, value in enumerate(values):
        value = str(value).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
        first_position.setdefault(value, position)
    if not counts:
        return ""
    return min(counts, key=lambda value: (-counts[value], first_position[value]))


def _extract_size(text: str) -> str:
    """擷取文字尺寸；非數字開頭（如 CDA 15A）時改為搜尋後方。"""
    stripped = str(text).strip()
    if not stripped:
        return ""
    if re.match(r"^\d", stripped):
        match = re.match(rf"(?P<size>{SIZE_TOKEN})(?=\s|[xX×]|$)", stripped, re.IGNORECASE)
    else:
        match = re.search(rf"(?P<size>{SIZE_TOKEN})(?=\s|[xX×]|$)", stripped, re.IGNORECASE)
    return _normalize_size(match.group("size")) if match else ""


def assign_nearest_text_to_groups(groups: Sequence[EntityGroup], texts: Sequence[dict]) -> dict[int, dict]:
    """一般元件只和含尺寸文字進行綜合分數配對。"""
    size_texts = []
    for text in texts:
        size = _extract_size(text["text"])
        if size:
            size_texts.append({**text, "size": size})

    result = {}
    for group in groups:
        best = _best_text_match(group.center, size_texts)
        if best:
            score, distance, text, _ = best
            result[group.group_id] = {**text, "score": score, "distance": distance}
    return result


def _components_for_nodes(adjacency: dict[int, set[int]], nodes: set[int]) -> list[list[int]]:
    unseen = set(nodes)
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, found = deque([start]), []
        while queue:
            node = queue.popleft()
            found.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(found)
    return components


def _is_size_boundary(group: EntityGroup) -> bool:
    return _canonical_name(group.name) in {"reducer", "reducer-union"}


def _is_gauge(group: EntityGroup) -> bool:
    canonical = _canonical_name(group.name)
    return "gauge" in canonical or "guage" in canonical


def _is_hose(group: EntityGroup) -> bool:
    return _canonical_name(group.name) in {"hose", "軟管"}


def _is_special_rtee(group: EntityGroup) -> bool:
    return _canonical_name(group.name) in {"union r.tee", "vcr r.tee"}


def fill_by_connectivity_vote(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
) -> tuple[dict[int, dict], dict[int, dict]]:
    """依完整連通塊投票材質，再依移除 Reducer 的連通塊投票尺寸。"""
    tube_descriptions = assign_tube_descriptions(groups, texts)
    nearest_texts = assign_nearest_text_to_groups(groups, texts)

    # 先保存每個一般圖元組最近的文字，Tube 則只採專用格式配對結果。
    for group in groups:
        if group.group_id in nearest_texts and group.name != "Tube":
            group.matched_text = nearest_texts[group.group_id]["text"]

    # 材質：在完整鄰接矩陣的每個連通塊內，依 Tube 描述投票表面處理。
    for connected in _connected_components(adjacency):
        ordered_descriptions = sorted(
            (tube_descriptions[node] for node in connected if node in tube_descriptions),
            key=lambda item: item["text_index"],
        )
        finish = majority_first([item["finish"] for item in ordered_descriptions])
        for node in connected:
            group = groups[node]
            canonical = _canonical_name(group.name)
            if group.name == "Tube":
                description = tube_descriptions.get(node)
                group.material = description["material"] if description else ""
            elif canonical in {"hose", "軟管"}:
                group.material = ""
            elif canonical == "gasket":
                group.material = "SUS"
            elif canonical in FINISH_MATERIAL_NAMES:
                group.material = f"SUS316L {finish}".strip()
            else:
                group.material = "SUS316L"

    # 尺寸：移除 Reducer 與 Reducer-Union 後重算連通塊。
    non_reducer_nodes = {group.group_id for group in groups if not _is_size_boundary(group)}
    size_components = _components_for_nodes(adjacency, non_reducer_nodes)
    node_uniform_size: dict[int, str] = {}
    for connected in size_components:
        indexed_size_votes = []
        for node in connected:
            group = groups[node]
            if group.color == TUBE_COLOR:
                # Tube 只能用合法 Tube 規格文字投票；未配對的輔助管線不投票。
                if node in tube_descriptions:
                    indexed_size_votes.append((
                        tube_descriptions[node]["text_index"],
                        tube_descriptions[node]["size"],
                    ))
                continue
            elif node in nearest_texts:
                size = nearest_texts[node]["size"]
                if size:
                    indexed_size_votes.append((nearest_texts[node]["index"], size))
        indexed_size_votes.sort(key=lambda item: item[0])
        size_votes = [value for _, value in indexed_size_votes]
        uniform_size = majority_first(size_votes)
        for node in connected:
            node_uniform_size[node] = uniform_size
            group = groups[node]
            if group.name == "Tube":
                description = tube_descriptions.get(node)
                if description:
                    group.name = description["name"]
                group.size = description["size"] if description else uniform_size
                group.quantity = description["length"] if description else 0.0
                group.matched_text = description["text"] if description else ""
            elif _is_gauge(group):
                group.size = '1/4"'
                group.quantity = 1.0
            elif _canonical_name(group.name) == "gasket" and any(
                _is_gauge(groups[neighbor]) for neighbor in adjacency[group.group_id]
            ):
                group.size = '1/4"'
                group.quantity = 1.0
            elif _is_special_rtee(group):
                group.size = f'{uniform_size}x1/4"' if uniform_size else 'x1/4"'
                group.quantity = 1.0
            elif _is_hose(group):
                nearest = _best_text_match(group.center, texts, HOSE_RE)
                if nearest:
                    _, _, text, match = nearest
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

    # Reducer/Reducer-Union 由相鄰兩側連通塊的尺寸組合。
    for group in groups:
        if not _is_size_boundary(group):
            continue
        neighbor_sizes = []
        for neighbor in sorted(adjacency[group.group_id]):
            size = node_uniform_size.get(neighbor, groups[neighbor].size)
            if size and size not in neighbor_sizes:
                neighbor_sizes.append(size)
        if len(neighbor_sizes) >= 2:
            group.size = f"{neighbor_sizes[0]}x{neighbor_sizes[1]}"
        else:
            nearest = _best_text_match(group.center, texts, REDUCER_RE)
            if nearest:
                _, _, text, match = nearest
                group.size = f'{_normalize_size(match.group("first"))}x{_normalize_size(match.group("second"))}'
                group.matched_text = text["text"]
            elif neighbor_sizes:
                group.size = neighbor_sizes[0]
                group.notes.append(f"{group.name} 只找到單側尺寸")
            else:
                group.size = ""
                group.notes.append(f"{group.name} 找不到相鄰尺寸")
        group.quantity = 1.0

    return tube_descriptions, nearest_texts


def make_reports(groups: Sequence[EntityGroup]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for group in groups:
        detail_rows.append({
            "群組編號": group.group_id,
            "品名": group.name,
            "顏色": group.color,
            "尺寸": group.size,
            "材質": group.material,
            "原始數量": group.quantity,
            "配對文字": group.matched_text,
            "中心 X": group.center[0],
            "中心 Y": group.center[1],
            "圖元數": len(group.geometries),
            "來源列號": ",".join(map(str, group.source_rows)),
            "備註": "；".join(group.notes),
        })
    details = pd.DataFrame(detail_rows)

    aggregated: dict[tuple[str, str, str], float] = {}
    for group in groups:
        if group.name == "IGNORE":
            continue
        key = (group.name, group.size, group.material)
        if group.name in {"Tube", "Coil Tube", "Hose", "軟管"}:
            # 沒有合法長度文字的圖元數量為 0，不以幾何長度臆測。
            if group.quantity <= 0:
                continue
            aggregated[key] = aggregated.get(key, 0.0) + group.quantity
        else:
            aggregated[key] = aggregated.get(key, 0.0) + 1.0

    bom_rows = []
    for (name, size, material), quantity in aggregated.items():
        display_quantity = f"{_format_number(quantity)}M" if name in {"Tube", "Coil Tube", "Hose", "軟管"} else int(quantity)
        bom_rows.append({"品名": name, "尺寸": size, "材質": material, "數量": display_quantity})
    return pd.DataFrame(bom_rows, columns=["品名", "尺寸", "材質", "數量"]), details


def build_network_bom(
    df: pd.DataFrame,
    component_tolerance: float = COMPONENT_TOLERANCE,
    connection_tolerance: float = CONNECTION_TOLERANCE,
    duplicate_tolerance: float = DUPLICATE_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups, removed_rows = build_entity_groups(df, component_tolerance, duplicate_tolerance)
    matrix, adjacency = build_adjacency_matrix(groups, connection_tolerance)
    texts = extract_texts(df)
    fill_by_connectivity_vote(groups, adjacency, texts)
    bom, details = make_reports(groups)
    details.attrs["removed_duplicate_rows"] = removed_rows
    labels = [f"G{group.group_id}:{group.name}" for group in groups]
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.index.name = "群組"
    return bom, details, matrix_df


def main(file_path: str) -> int:
    """讀取指定 DXF/DWG/CSV，並在輸入檔旁產生 BOM、明細及鄰接矩陣。"""
    if COMPONENT_TOLERANCE < 0 or CONNECTION_TOLERANCE < 0 or DUPLICATE_TOLERANCE < 0:
        raise ValueError("容許距離不可小於 0")
    input_path = Path(file_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

    output = input_path.with_name(f"{input_path.stem}{BOM_OUTPUT_SUFFIX}")
    details_output = input_path.with_name(f"{input_path.stem}{DETAILS_OUTPUT_SUFFIX}")
    matrix_output = input_path.with_name(f"{input_path.stem}{ADJACENCY_OUTPUT_SUFFIX}")
    df = load_vector_data(input_path)
    bom, details, matrix = build_network_bom(
        df,
        component_tolerance=COMPONENT_TOLERANCE,
        connection_tolerance=CONNECTION_TOLERANCE,
        duplicate_tolerance=DUPLICATE_TOLERANCE,
    )
    removed_rows = details.attrs.get("removed_duplicate_rows", [])

    for path in (output, details_output, matrix_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    bom.to_csv(output, index=False, encoding="utf-8-sig")
    details.to_csv(details_output, index=False, encoding="utf-8-sig")
    matrix.to_csv(matrix_output, encoding="utf-8-sig")
    print(f"BOM：{output}（{len(bom)} 筆）")
    print(f"群組明細：{details_output}（{len(details)} 組）")
    print(f"鄰接矩陣：{matrix_output}（{len(matrix)} x {len(matrix)}）")
    print(f"去除同色重複線段：{len(removed_rows)} 條（原始列：{removed_rows}）")
    return 0


if __name__ == "__main__":
    import os
    for path in os.listdir("./data/data_new/"):
        if path.lower().endswith((".dxf", ".dwg")):
            print(f"處理檔案：{path}")
            main(os.path.join("./data/data_new/", path))
