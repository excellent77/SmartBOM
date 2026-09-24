"""DXF 文字擷取、規格解析與最近文字配對。"""

import math
import re
from typing import Sequence
import pandas as pd
from .config import TEXT_TYPES, TUBE_COLOR
from .models import EntityGroup
from .predicates import is_size_boundary
from .geometry import _number, _point

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
        if group.color == TUBE_COLOR or is_size_boundary(group):
            continue
        best = _best_text_match(group.center, size_texts)
        if best:
            distance, text, _ = best
            result[group.group_id] = {**text, "distance": distance}
    return result



class TextSpecificationService:
    """集中提供文字正規化、規格解析及元件文字配對。"""
    extract = staticmethod(extract_texts)
    parse_tube = staticmethod(_tube_description)
    normalize_size = staticmethod(_normalize_size)
    nearest_for_groups = staticmethod(assign_nearest_text_to_groups)
