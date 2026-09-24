"""把最佳化結果轉換成 BOM 欄位的工程領域規則。"""

from typing import Sequence
from .models import EntityGroup
from .grouping import _connected_components
from .optimizer import TubeAssignmentOptimizer
from .predicates import canonical_name, is_gauge, is_hose, is_size_boundary, is_special_rtee
from .text import FINISH_MATERIAL_NAMES, HOSE_RE, REDUCER_RE, _best_text_match, _normalize_size, assign_nearest_text_to_groups, majority_first

def fill_by_integer_programming(
    groups: Sequence[EntityGroup],
    adjacency: dict[int, set[int]],
    texts: Sequence[dict],
    optimizer: TubeAssignmentOptimizer | None = None,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """以 ILP 決定 Tube 配對及分段尺寸，再套用既有特殊元件規則。"""
    nearest_texts = assign_nearest_text_to_groups(groups, texts)
    assignment_service = optimizer or TubeAssignmentOptimizer()
    tube_descriptions, node_uniform_size = assignment_service.assign(
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
            canonical = canonical_name(group.name)
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
        if is_size_boundary(group):
            continue
        uniform_size = node_uniform_size.get(group.group_id, "")
        if group.name == "Tube":
            description = tube_descriptions.get(group.group_id)
            if description:
                group.name = description["name"]
            group.size = uniform_size
            group.quantity = description["length"] if description else 0.0
            group.matched_text = description["text"] if description else ""
        elif is_gauge(group):
            group.size = '1/4"'
            group.quantity = 1.0
        elif canonical_name(group.name) == "gasket" and any(
            is_gauge(groups[neighbor]) for neighbor in adjacency[group.group_id]
        ):
            group.size = '1/4"'
            group.quantity = 1.0
        elif is_special_rtee(group):
            group.size = f'{uniform_size}x1/4"' if uniform_size else 'x1/4"'
            group.quantity = 1.0
        elif is_hose(group):
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
        if not is_size_boundary(group):
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



class BomRuleEngine:
    """套用材質、尺寸、Gauge、Hose 與 Reducer 等規則。"""
    def __init__(self, optimizer: TubeAssignmentOptimizer | None = None):
        self.optimizer = optimizer or TubeAssignmentOptimizer()

    def apply(self, groups, adjacency, texts):
        return fill_by_integer_programming(groups, adjacency, texts, self.optimizer)
