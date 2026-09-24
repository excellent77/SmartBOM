"""幾何轉換、距離、聚類與重複線段消除。"""

import math
from typing import Sequence
import numpy as np
import pandas as pd
from .config import ARC_SAMPLE_STEPS, CIRCLE_SAMPLE_STEPS, GEOMETRY_TYPES, MIN_ARC_SAMPLE_STEPS, SEGMENT_INTERSECTION_EPSILON
from .models import Geometry

class DisjointSet:
    """供幾何連通聚類使用的 disjoint-set 資料結構。"""

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



class GeometryService:
    """提供可注入的幾何操作門面，隱藏底層數值函式。"""
    row_to_geometry = staticmethod(row_to_geometry)
    distance = staticmethod(geometry_distance)
    cluster = staticmethod(cluster_geometries)
    remove_duplicates = staticmethod(remove_duplicate_lines)
    center = staticmethod(component_center)
