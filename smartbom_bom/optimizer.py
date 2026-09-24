"""Tube 文字分配與連通段尺寸選擇的二元整數規劃。"""

import math
from typing import Sequence
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as ilp
from scipy.sparse import lil_matrix
from .config import COMPONENT_SIZE_SCORE_WEIGHT, TUBE_COLOR
from .models import EntityGroup
from .grouping import _components_for_nodes
from .predicates import is_size_boundary
from .text import _tube_description

def assign_tube_descriptions(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
    component_text_matches: dict[int, dict],
    size_score_weight: float = COMPONENT_SIZE_SCORE_WEIGHT,
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
    non_boundary_nodes = {group.group_id for group in groups if not is_size_boundary(group)}
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
        objective[y_index[key]] = -count * size_score_weight

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



class TubeAssignmentOptimizer:
    """可替換的 Tube 指派服務；目前以 SciPy MILP 實作。"""
    def __init__(self, size_score_weight: float = COMPONENT_SIZE_SCORE_WEIGHT):
        self.size_score_weight = size_score_weight

    def assign(self, groups, adjacency, texts, component_text_matches):
        return assign_tube_descriptions(
            groups, adjacency, texts, component_text_matches, self.size_score_weight
        )
