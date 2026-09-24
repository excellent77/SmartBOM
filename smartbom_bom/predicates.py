"""元件名稱判定規則；其他模組只依賴這個穩定介面。"""

import re
from .models import EntityGroup

def canonical_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().casefold())

def is_size_boundary(group: EntityGroup) -> bool:
    return canonical_name(group.name) in {"reducer", "reducer-union"}


def is_gauge(group: EntityGroup) -> bool:
    canonical = canonical_name(group.name)
    return "gauge" in canonical or "guage" in canonical


def is_hose(group: EntityGroup) -> bool:
    return canonical_name(group.name) in {"hose", "軟管"}


def is_special_rtee(group: EntityGroup) -> bool:
    return canonical_name(group.name) in {"union r.tee", "vcr r.tee"}

