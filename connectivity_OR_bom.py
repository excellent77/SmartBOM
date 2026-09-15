"""以去重、圖元組鄰接關係及二元整數線性規劃（ILP）產生工程 BOM。

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
from scipy.optimize import Bounds, LinearConstraint, milp as ilp
from scipy.sparse import lil_matrix

from dxf_reader import read_dwg


# ===== 可調整參數 =====
COMPONENT_TOLERANCE = 0.5
CONNECTION_TOLERANCE = 0.5
DUPLICATE_TOLERANCE = 0.01
SEGMENT_INTERSECTION_EPSILON = 1e-9
CIRCLE_SAMPLE_STEPS = 72
ARC_SAMPLE_STEPS = 36
MIN_ARC_SAMPLE_STEPS = 8
COMPONENT_SIZE_SCORE_WEIGHT = 1.0
TUBE_COLOR = "顏色_30"
BOM_OUTPUT_SUFFIX = "_bom.csv"
DETAILS_OUTPUT_SUFFIX = "_details.csv"
ADJACENCY_OUTPUT_SUFFIX = "_adjacency.csv"


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
    "顏色_7": {"name": "IGNORE"},  # 表上註記「不須列入計算的材料」，應在後製時直接排除，不當成 BOM 品項
}

TEXT_TYPES = {"文字", "多行文字", "TEXT", "MTEXT"}
GEOMETRY_TYPES = {"線", "聚合線", "圓", "弧", "橢圓", "填充線", "LINE", "CIRCLE", "ARC", "ELLIPSE"}


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
    start = _point(row, ("起點 X",), ("起點 Y",))
    end = _point(row, ("終點 X",), ("終點 Y",))
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
    kind, color = str(row.get("名稱", "")).strip(), str(row.get("出圖型式", "")).strip()
    if kind not in GEOMETRY_TYPES or not color:
        return None
    start = _point(row, ("起點 X",), ("起點 Y",))
    end = _point(row, ("終點 X",), ("終點 Y",))
    center = _point(row, ("中心點 X",), ("中心點 Y",))
    radius = _number(row, "半徑")
    if kind in {"線", "聚合線", "LINE"} and start and end:
        points = np.asarray([start, end], dtype=float)
    elif kind in {"圓", "CIRCLE"} and center and radius is not None:
        points = _sample_circle(center, abs(radius))
    elif kind in {"弧", "ARC"} and center and radius is not None:
        points = _sample_arc(row, center, abs(radius))
    elif kind == "填充線" and start and end:
        x1, y1 = start
        x2, y2 = end
        points = np.asarray([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dtype=float)
    elif kind in {"橢圓", "ELLIPSE"} and center:
        points = np.asarray([center], dtype=float)
    else:
        return None
    return Geometry(index, color, kind, points)


def clean_dxf_text(value: str) -> str:
    text = str(value).replace("\\P", " ").replace("\\~", " ").replace("^I", " ")
    text = re.sub(r"\\[AaCcFfHhQqTtWw][^;]*;", "", text)
    text = re.sub(r"\\[LlOoKk]", "", text).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def extract_texts(df: pd.DataFrame) -> list[dict]:
    texts = []
    for index, row in df.iterrows():
        if str(row.get("名稱", "")).strip() not in TEXT_TYPES:
            continue
        position = _point(row, ("位置 X", "位置 X1", "X"), ("位置 Y", "位置 Y1", "Y"))
        if position is None:
            continue
        value = ""
        for column in ("值", "內容"):
            candidate = row.get(column)
            if pd.notna(candidate) and str(candidate).strip():
                value = clean_dxf_text(candidate)
                break
        if value:
            texts.append({
                "index": int(index),
                "text": value,
                "position": position,
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


def _best_text_match(point: tuple[float, float], texts: Sequence[dict], pattern: re.Pattern | None = None):
    """依歐氏距離選最近文字；同距離時選原始列號較小者。"""
    candidates = []
    for text in texts:
        match = pattern.search(text["text"]) if pattern else None
        if pattern is not None and match is None:
            continue
        distance = math.dist(point, text["position"])
        candidates.append((distance, text, match))
    return min(candidates, key=lambda item: (item[0], item[1]["index"])) if candidates else None


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


def assign_tube_descriptions(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
    component_text_matches: dict[int, dict],
) -> tuple[dict[int, dict], dict[int, str]]:
    """以單次 ILP 同時決定 Tube 文字配對與各連通段代表尺寸。

    最小化總配對距離減去一般元件文字的尺寸支持分數；同一連通段配對到的
    Tube 必須採用相同的 ``(尺寸, 材質與表面處理)`` 規格。
    每段合法 Tube 文字必須恰好配對一次，每個 Tube 至多接受一段文字；
    若無法滿足全部配對與規格限制，回報無可行解。
    """
    tube_groups = [group for group in groups if group.color == TUBE_COLOR]
    pending = [
        (text, description)
        for text in texts
        if (description := _tube_description(text["text"])) is not None
    ]
    if not tube_groups:
        if pending:
            raise RuntimeError("Tube 文字整數規劃無可行解：存在合法 Tube 規格文字，但沒有 Tube 圖元可配對")
        return {}, {}

    # Reducer 類元件不參與分段。
    non_boundary_nodes = {group.group_id for group in groups if not _is_size_boundary(group)}
    segments = _components_for_nodes(adjacency, non_boundary_nodes)
    node_segment = {node: segment_id for segment_id, nodes in enumerate(segments) for node in nodes}
    active_segments = list(range(len(segments)))

    def spec_key(description: dict) -> tuple[str, str]:
        return description["size"], description["material"]

    specs = sorted({spec_key(description) for _, description in pending})
    sizes = sorted(
        {description["size"] for _, description in pending}
        | {
            match["size"]
            for group_id, match in component_text_matches.items()
            if group_id in node_segment and match.get("size")
        }
    )
    if not sizes:
        return {}, {}

    tube_ids = [group.group_id for group in tube_groups]
    tube_by_id = {group.group_id: group for group in tube_groups}

    # x：文字與 Tube；y：連通段代表尺寸；z：連通段完整 Tube 規格。
    x_pairs = [(text_id, tube_id) for text_id in range(len(pending)) for tube_id in tube_ids]
    x_index = {pair: position for position, pair in enumerate(x_pairs)}
    y_offset = len(x_index)
    y_pairs = [(segment_id, size) for segment_id in active_segments for size in sizes]
    y_index = {pair: y_offset + position for position, pair in enumerate(y_pairs)}
    z_offset = y_offset + len(y_index)
    z_pairs = [(segment_id, spec) for segment_id in active_segments for spec in specs]
    z_index = {pair: z_offset + position for position, pair in enumerate(z_pairs)}
    variable_count = len(x_index) + len(y_index) + len(z_index)
    objective = np.zeros(variable_count, dtype=float)
    distance_cache: dict[tuple[int, int], float] = {}
    for text_id, (text, _description) in enumerate(pending):
        for tube_id in tube_ids:
            distance = math.dist(tube_by_id[tube_id].center, text["position"])
            distance_cache[text_id, tube_id] = distance
            objective[x_index[text_id, tube_id]] = distance

    size_occurrences: dict[tuple[int, str], int] = {}
    for group_id, match in component_text_matches.items():
        if group_id not in node_segment or not match.get("size"):
            continue
        key = node_segment[group_id], match["size"]
        size_occurrences[key] = size_occurrences.get(key, 0) + 1
    for key, count in size_occurrences.items():
        objective[y_index[key]] = -count * COMPONENT_SIZE_SCORE_WEIGHT

    constraint_rows: list[dict[int, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(coefficients: dict[int, float], lower: float, upper: float) -> None:
        constraint_rows.append(coefficients)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # 每段合法 Tube 文字恰好配對一次，每條 Tube 至多接受一段文字。
    for text_id in range(len(pending)):
        add_constraint({x_index[text_id, tube_id]: 1.0 for tube_id in tube_ids}, 1.0, 1.0)
    for tube_id in tube_ids:
        add_constraint({x_index[text_id, tube_id]: 1.0 for text_id in range(len(pending))}, 0.0, 1.0)

    # 每段恰選一個代表尺寸，並至多採用一種完整 Tube 規格。
    for segment_id in active_segments:
        add_constraint({y_index[segment_id, size]: 1.0 for size in sizes}, 1.0, 1.0)
        if specs:
            add_constraint({z_index[segment_id, spec]: 1.0 for spec in specs}, 0.0, 1.0)

    # 完整 Tube 規格必須服從該段代表尺寸。
    for segment_id in active_segments:
        for spec in specs:
            add_constraint(
                {z_index[segment_id, spec]: 1.0, y_index[segment_id, spec[0]]: -1.0},
                -np.inf,
                0.0,
            )

    # Tube 文字必須服從該段唯一的完整 Tube 規格。
    for text_id, (_text, description) in enumerate(pending):
        spec = spec_key(description)
        for tube_id in tube_ids:
            segment_id = node_segment[tube_id]
            add_constraint(
                {x_index[text_id, tube_id]: 1.0, z_index[segment_id, spec]: -1.0},
                -np.inf,
                0.0,
            )

    matrix = lil_matrix((len(constraint_rows), variable_count), dtype=float)
    for row_id, coefficients in enumerate(constraint_rows):
        for variable_id, coefficient in coefficients.items():
            matrix[row_id, variable_id] = coefficient
    base_constraint = LinearConstraint(matrix.tocsr(), lower_bounds, upper_bounds)

    # 所有變數均為 0–1；使用 SciPy 的通用 MILP 求解器求解此 ILP。
    result = ilp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=base_constraint,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Tube 文字整數規劃無法求解：{result.message}")

    segment_sizes = {
        segment_id: next(size for size in sizes if result.x[y_index[segment_id, size]] > 0.5)
        for segment_id in active_segments
    }
    assignments: dict[int, dict] = {}
    for text_id, (text, description) in enumerate(pending):
        for tube_id in tube_ids:
            if result.x[x_index[text_id, tube_id]] <= 0.5:
                continue
            distance = distance_cache[text_id, tube_id]
            assignments[tube_id] = {
                **description,
                "text": text["text"],
                "text_index": text["index"],
                "distance": distance,
            }
    node_sizes = {
        node: segment_sizes[segment_id]
        for node, segment_id in node_segment.items()
        if segment_id in segment_sizes
    }
    return assignments, node_sizes


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
    """一般元件配對含尺寸文字，排除 Tube 規格與雙尺寸 Reducer 文字。"""
    size_texts = []
    for text in texts:
        if _tube_description(text["text"]) is not None or REDUCER_RE.search(text["text"]):
            continue
        size = _extract_size(text["text"])
        if size:
            size_texts.append({**text, "size": size})

    result = {}
    for group in groups:
        if group.color == TUBE_COLOR or _is_size_boundary(group):
            continue
        best = _best_text_match(group.center, size_texts)
        if best:
            distance, text, _ = best
            result[group.group_id] = {**text, "distance": distance}
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


def fill_by_integer_programming(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
) -> tuple[dict[int, dict], dict[int, dict]]:
    """以 ILP 決定 Tube 配對及分段尺寸，再套用既有特殊元件規則。"""
    nearest_texts = assign_nearest_text_to_groups(groups, texts)
    tube_descriptions, node_uniform_size = assign_tube_descriptions(
        groups, adjacency, texts, nearest_texts
    )

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

    # 尺寸直接使用 ILP 選出的連通段代表尺寸，不再進行眾數投票。
    for group in groups:
        if _is_size_boundary(group):
            continue
        uniform_size = node_uniform_size.get(group.group_id, "")
        if group.name == "Tube":
            description = tube_descriptions.get(group.group_id)
            if description:
                group.name = description["name"]
            group.size = uniform_size
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
                _, text, match = nearest
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
    fill_by_integer_programming(groups, adjacency, texts)
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
