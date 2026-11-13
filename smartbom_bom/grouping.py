"""將幾何圖元組成元件，並建立元件間的拓撲圖。"""

from collections import deque
from typing import Sequence
import numpy as np
import pandas as pd
from .config import COLOR_COMPONENT_MAP, COMPONENT_TYPE_TUBE, TUBE_COLOR
from .models import EntityGroup, Geometry
from .geometry import component_center, geometry_distance, cluster_geometries, remove_duplicate_lines, row_to_geometry
from .predicates import infer_component_type

def build_entity_groups(
    df: pd.DataFrame,
    tolerance: float,
    duplicate_tolerance: float,
    tube_color: str = TUBE_COLOR,
    component_map: dict[str, dict[str, str]] | None = None,
) -> tuple[list[EntityGroup], list[int]]:
    component_map = COLOR_COMPONENT_MAP if component_map is None else component_map
    geometries = [geometry for index, row in df.iterrows() if (geometry := row_to_geometry(int(index), row))]
    geometries, removed_rows = remove_duplicate_lines(geometries, duplicate_tolerance)
    def definition(color: str) -> tuple[str, str]:
        configured = component_map.get(color, {})
        name = configured.get("name", color)
        component_type = configured.get("type") or (
            COMPONENT_TYPE_TUBE if color == tube_color else infer_component_type(name, color)
        )
        return name, component_type

    tubes = [
        geometry for geometry in geometries
        if definition(geometry.color)[1] == COMPONENT_TYPE_TUBE
    ]
    components = [
        geometry for geometry in geometries
        if definition(geometry.color)[1] != COMPONENT_TYPE_TUBE
    ]

    raw_groups: list[tuple[str, str, str, list[Geometry]]] = []
    # 使用者標示為 Tube 類的每個圖元本身就是一組。
    raw_groups.extend((*definition(geometry.color), geometry.color, [geometry]) for geometry in tubes)
    for cluster in cluster_geometries(components, tolerance):
        color = cluster[0].color
        component_name, component_type = definition(color)
        raw_groups.append((component_name, component_type, color, cluster))

    groups = []
    for group_id, (name, component_type, color, items) in enumerate(raw_groups):
        groups.append(EntityGroup(
            group_id=group_id,
            name=name,
            color=color,
            geometries=items,
            center=component_center(items),
            source_rows=[item.row_index for item in items],
            component_type=component_type,
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



class EntityGroupingService:
    """負責元件聚類、去重與鄰接圖建構。"""
    def __init__(self, tube_color: str = TUBE_COLOR, component_map=None):
        self.tube_color = tube_color
        self.component_map = COLOR_COMPONENT_MAP if component_map is None else component_map

    def build_groups(self, df: pd.DataFrame, tolerance: float, duplicate_tolerance: float):
        return build_entity_groups(
            df, tolerance, duplicate_tolerance, self.tube_color, self.component_map
        )
    def build_adjacency(self, groups: Sequence[EntityGroup], tolerance: float):
        return build_adjacency_matrix(groups, tolerance)
