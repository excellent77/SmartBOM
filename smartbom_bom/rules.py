"""把最佳化結果轉換成 BOM 欄位的工程領域規則。"""

from typing import Sequence
from .models import EntityGroup
from .grouping import _connected_components
from .optimizer import TubeAssignmentOptimizer
from .predicates import (
    canonical_name, has_fixed_quarter_inch_size, is_hose, is_size_boundary, is_tube,
)
from .text import FINISH_MATERIAL_NAMES, HOSE_RE, _best_text_match, _normalize_size, assign_nearest_text_to_groups, majority_first

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
        if group.group_id in nearest_texts and not is_tube(group):
            group.matched_text = nearest_texts[group.group_id]["text"]

    # 材質：在完整鄰接矩陣的每個連通塊內，依 Tube 描述同時投票
    # 基本材質與表面處理。需表面處理的管件會跟隨管線的 SUS304L／SUS316L；
    # 其他一般元件（包含 Gasket）固定為 SUS316L，不再產生模糊的 SUS。
    for connected in _connected_components(adjacency):
        ordered_descriptions = sorted(
            (tube_descriptions[node] for node in connected if node in tube_descriptions),
            key=lambda item: item["text_index"],
        )
        base_material = majority_first([
            str(item["base_material"]).upper() for item in ordered_descriptions
        ]) or "SUS316L"
        finish = majority_first([item["finish"] for item in ordered_descriptions])
        for node in connected:
            group = groups[node]
            canonical = canonical_name(group.name)
            if is_tube(group):
                description = tube_descriptions.get(node)
                group.material = description["material"] if description else ""
            elif canonical in {"hose", "軟管"}:
                group.material = ""
            elif is_size_boundary(group) or canonical in FINISH_MATERIAL_NAMES:
                group.material = f"{base_material} {finish}".strip()
            else:
                group.material = "SUS316L"

    # 尺寸直接使用 ILP 選出的連通段代表尺寸，不再進行眾數投票。
    for group in groups:
        if is_size_boundary(group):
            continue
        uniform_size = node_uniform_size.get(group.group_id, "")
        if is_tube(group):
            description = tube_descriptions.get(group.group_id)
            if description:
                group.name = description["name"]
            group.size = uniform_size
            group.quantity = description["length"] if description else 0.0
            group.matched_text = description["text"] if description else ""
        elif has_fixed_quarter_inch_size(group, groups, adjacency):
            group.size = '1/4"'
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

    # 所有 Reducer 類由全部相鄰方向的連通段尺寸組合。
    # 可以連接二、三或更多方向，但最終必須剛好只有兩種尺寸。
    for group in groups:
        if not is_size_boundary(group):
            continue
        neighbor_sizes = []
        for neighbor in sorted(adjacency[group.group_id]):
            neighbor_group = groups[neighbor]
            # Gauge 固定尺寸必須先於連通段代表尺寸生效，否則 R.Tee 會把
            # Gauge 所在連通段的主尺寸誤當成 Gauge 方向尺寸。
            size = (
                '1/4"'
                if has_fixed_quarter_inch_size(neighbor_group, groups, adjacency)
                else node_uniform_size.get(neighbor, neighbor_group.size)
            )
            if size and size not in neighbor_sizes:
                neighbor_sizes.append(size)
        if len(neighbor_sizes) == 2:
            group.size = f"{neighbor_sizes[0]}x{neighbor_sizes[1]}"
        else:
            found = "、".join(neighbor_sizes) if neighbor_sizes else "無"
            group.size = ""
            group.notes.append(
                f"G{group.group_id}:{group.name} 為 Reducer類，但所有連接方向"
                f"不是剛好兩種尺寸（找到：{found}）"
            )
        group.quantity = 1.0

    return tube_descriptions, nearest_texts



class BomRuleEngine:
    """套用材質、尺寸、Gauge、Hose 與 Reducer 等規則。"""
    def __init__(self, optimizer: TubeAssignmentOptimizer | None = None):
        self.optimizer = optimizer or TubeAssignmentOptimizer()

    def apply(self, groups, adjacency, texts):
        return fill_by_integer_programming(groups, adjacency, texts, self.optimizer)
