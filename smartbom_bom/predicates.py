"""元件名稱判定規則；其他模組只依賴這個穩定介面。"""

import re
from collections.abc import Mapping, Sequence
from .config import (
    COMPONENT_TYPE_GENERAL, COMPONENT_TYPE_REDUCER, COMPONENT_TYPE_TUBE, TUBE_COLOR,
)
from .models import EntityGroup

def canonical_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().casefold())


def infer_component_type(name: str, color: str = "") -> str:
    """在舊映射尚無 type 欄位時，依名稱與傳統 Tube 色提供相容預設。"""
    canonical = canonical_name(name)
    if color == TUBE_COLOR or canonical in {"tube", "coil tube", "over tube"}:
        return COMPONENT_TYPE_TUBE
    if canonical in {
        "reducer", "reducer-union", "reducer union",
        "r.tee", "r tee", "union r.tee", "union r tee",
        "vcr r.tee", "vcr r tee",
    }:
        return COMPONENT_TYPE_REDUCER
    return COMPONENT_TYPE_GENERAL


def is_tube(group: EntityGroup) -> bool:
    return (
        group.component_type == COMPONENT_TYPE_TUBE
        or (
            not group.component_type
            and infer_component_type(group.name, group.color) == COMPONENT_TYPE_TUBE
        )
    )


def is_size_boundary(group: EntityGroup) -> bool:
    return (
        group.component_type == COMPONENT_TYPE_REDUCER
        or (
            not group.component_type
            and infer_component_type(group.name, group.color) == COMPONENT_TYPE_REDUCER
        )
    )


def is_gauge(group: EntityGroup) -> bool:
    canonical = canonical_name(group.name)
    return "gauge" in canonical or "guage" in canonical


def has_fixed_quarter_inch_size(
    group: EntityGroup,
    groups: Sequence[EntityGroup],
    adjacency: Mapping[int, set[int]],
) -> bool:
    """Gauge 與直接連接 Gauge 的 Gasket 一律視為 1/4 吋。"""
    if is_gauge(group):
        return True
    return canonical_name(group.name) == "gasket" and any(
        is_gauge(groups[neighbor]) for neighbor in adjacency.get(group.group_id, set())
    )


def is_hose(group: EntityGroup) -> bool:
    return canonical_name(group.name) in {"hose", "軟管"}

