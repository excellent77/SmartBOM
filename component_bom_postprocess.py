"""讀取 DXF/DWG，透過 dxf_reader 取得向量資料並聚類成元件 BOM。

範例：
    python component_bom_postprocess.py data/16_單線圖_KOXDLP2000.dxf
    python component_bom_postprocess.py input.dwg -o output_bom.csv --tolerance 0.5

流程：同色幾何分組 -> 依接觸、相交、重疊或近接聚類 -> 計算中心 ->
以距離和元件名稱關鍵字替每個元件選文字 -> 彙總 BOM。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from dxf_reader import read_dwg


# 可直接修改這個 dict。key 必須和 CSV 的「出圖型式」一致。
# name 是 BOM 元件名；english/chinese 會用於文字配對加分。
COLOR_COMPONENT_MAP = {
    "顏色_181": {"name": "Ball Valve", "english": ["ball valve", "b.v"], "chinese": ["球閥"]},
    "顏色_3": {"name": "Diaphragm Valve", "english": ["diaphragm valve", "d.v"], "chinese": ["隔膜閥"]},
    "顏色_212": {"name": "Check Valve", "english": ["check valve", "c.v"], "chinese": ["止回閥", "逆止閥"]},
    "顏色_6": {"name": "Regulator", "english": ["regulator"], "chinese": ["調壓閥", "調節閥"]},
    "顏色_140": {"name": "3P Regulator", "english": ["3p regulator"], "chinese": ["三點式調壓閥"]},
    "顏色_134": {"name": "Tee", "english": ["tee"], "chinese": ["三通"]},
    "顏色_5": {"name": "Reducer", "english": ["reducer"], "chinese": ["異徑", "大小頭"]},
    "顏色_16": {"name": "Elbow", "english": ["elbow"], "chinese": ["彎頭"]},
    "顏色_1": {"name": "NUT(F)", "english": ["nut(f)", "female nut"], "chinese": ["母螺帽"]},
    "顏色_8": {"name": "NUT(M)", "english": ["nut(m)", "male nut"], "chinese": ["公螺帽"]},
    "顏色_142": {"name": "S Gland", "english": ["s gland"], "chinese": ["短壓蓋"]},
    "顏色_11": {"name": "L Gland", "english": ["l gland"], "chinese": ["長壓蓋"]},
    "顏色_165": {"name": "Gasket", "english": ["gasket"], "chinese": ["墊片"]},
    "顏色_7": {"name": "VCR Tee", "english": ["vcr tee"], "chinese": ["vcr 三通"]},
    "顏色_84": {"name": "VCR Union Tee", "english": ["vcr union tee"], "chinese": ["vcr 正三通"]},
    "顏色_171": {"name": "Union R.Tee", "english": ["union r.tee", "union reducing tee"], "chinese": ["異徑聯管三通"]},
    "顏色_241": {"name": "Union Tee", "english": ["union tee"], "chinese": ["聯管三通"]},
    "顏色_40": {"name": "Gauge", "english": ["gauge"], "chinese": ["壓力錶", "儀表"]},
    "顏色_67": {"name": "Union", "english": ["union"], "chinese": ["由令", "聯管"]},
    "顏色_211": {"name": "Reducer Union", "english": ["reducer union"], "chinese": ["異徑由令"]},
    "顏色_4": {"name": "Hose", "english": ["hose"], "chinese": ["軟管"]},
    "顏色_37": {"name": "Cap", "english": ["cap"], "chinese": ["管帽", "封頭"]},
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
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
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
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_vector_csv(path: str | Path) -> pd.DataFrame:
    """讀取 dxf_reader CSV；依序嘗試常見繁中編碼。"""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("無法判斷 CSV 編碼：" + "; ".join(errors))


def load_vector_data(path: str | Path) -> pd.DataFrame:
    """直接處理 DXF/DWG；CSV 僅保留作為除錯與舊流程的相容輸入。"""
    input_path = Path(path)
    extension = input_path.suffix.casefold()
    if extension in {".dxf", ".dwg"}:
        return read_dwg(str(input_path))
    if extension == ".csv":
        return read_vector_csv(input_path)
    raise ValueError(f"不支援的輸入格式：{extension}；請使用 .dxf、.dwg 或 .csv")


def _number(row: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def _point(row: pd.Series, x_columns: Sequence[str], y_columns: Sequence[str]) -> tuple[float, float] | None:
    x, y = _number(row, *x_columns), _number(row, *y_columns)
    return None if x is None or y is None else (x, y)


def _sample_circle(center: tuple[float, float], radius: float, steps: int = 72) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, steps + 1)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def _sample_arc(row: pd.Series, center: tuple[float, float], radius: float, steps: int = 36) -> np.ndarray:
    start = _point(row, ("起點 X",), ("起點 Y",))
    end = _point(row, ("終點 X",), ("終點 Y",))
    if start and end:
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    else:
        start_deg = _number(row, "起始角度", "角度1")
        total_deg = _number(row, "總角度")
        if start_deg is None or total_deg is None:
            return _sample_circle(center, radius, steps * 2)
        start_angle = math.radians(start_deg)
        end_angle = start_angle + math.radians(total_deg)
    while end_angle < start_angle:
        end_angle += 2.0 * math.pi
    count = max(8, int(steps * (end_angle - start_angle) / (2.0 * math.pi)))
    angles = np.linspace(start_angle, end_angle, count + 1)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def row_to_geometry(index: int, row: pd.Series) -> Geometry | None:
    kind, color = str(row.get("名稱", "")).strip(), str(row.get("出圖型式", "")).strip()
    if not kind or not color or kind not in GEOMETRY_TYPES:
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
    elif kind in {"填充線"} and start and end:
        x1, y1 = start
        x2, y2 = end
        points = np.asarray([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dtype=float)
    elif kind in {"橢圓", "ELLIPSE"} and center:
        # 目前 dxf_reader 只輸出橢圓中心；視為點，仍可和附近圖元聚合。
        points = np.asarray([center], dtype=float)
    else:
        return None
    return Geometry(index, color, kind, points)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = min(1.0, max(0.0, float(np.dot(point - start, delta) / length2)))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, epsilon: float = 1e-9) -> bool:
    ab, cd = b - a, d - c
    c1, c2, c3, c4 = _cross(ab, c - a), _cross(ab, d - a), _cross(cd, a - c), _cross(cd, b - c)
    if ((c1 > epsilon and c2 < -epsilon) or (c1 < -epsilon and c2 > epsilon)) and ((c3 > epsilon and c4 < -epsilon) or (c3 < -epsilon and c4 > epsilon)):
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
    p, q = first.points, second.points
    if len(p) == 1 and len(q) == 1:
        return float(np.linalg.norm(p[0] - q[0]))
    p_segments = list(zip(p[:-1], p[1:])) if len(p) > 1 else [(p[0], p[0])]
    q_segments = list(zip(q[:-1], q[1:])) if len(q) > 1 else [(q[0], q[0])]
    best = math.inf
    for a, b in p_segments:
        for c, d in q_segments:
            best = min(best, _segment_distance(a, b, c, d))
            if best <= stop_at:
                return best
    return best


def _bbox_near(first: Geometry, second: Geometry, tolerance: float) -> bool:
    a, b = first.bbox, second.bbox
    return not (a[2] + tolerance < b[0] or b[2] + tolerance < a[0] or a[3] + tolerance < b[1] or b[3] + tolerance < a[1])


def cluster_geometries(geometries: Sequence[Geometry], tolerance: float) -> list[list[Geometry]]:
    """以圖元相交、重疊或最短距離小於 tolerance 建立連通分量。"""
    groups: list[list[Geometry]] = []
    by_color: dict[str, list[Geometry]] = {}
    for geometry in geometries:
        by_color.setdefault(geometry.color, []).append(geometry)

    for color_geometries in by_color.values():
        dsu = DisjointSet(len(color_geometries))
        for left in range(len(color_geometries)):
            for right in range(left + 1, len(color_geometries)):
                first, second = color_geometries[left], color_geometries[right]
                if _bbox_near(first, second, tolerance) and geometry_distance(first, second, tolerance) <= tolerance:
                    dsu.union(left, right)
        color_groups: dict[int, list[Geometry]] = {}
        for index, geometry in enumerate(color_geometries):
            color_groups.setdefault(dsu.find(index), []).append(geometry)
        groups.extend(color_groups.values())
    return groups


def component_center(component: Sequence[Geometry]) -> tuple[float, float]:
    """以所有取樣點的外包框中心作元件中心，避免長線的密集取樣造成偏移。"""
    points = np.vstack([geometry.points for geometry in component])
    return (float((points[:, 0].min() + points[:, 0].max()) / 2), float((points[:, 1].min() + points[:, 1].max()) / 2))


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
                value = str(candidate).strip()
                break
        if value:
            texts.append({"index": int(index), "text": clean_dxf_text(value), "position": position})
    return texts


def clean_dxf_text(value: str) -> str:
    """移除舊版 CSV 常見的 MTEXT 控制碼，保留人可讀的規格內容。"""
    text = str(value).replace("\\P", " ").replace("\\~", " ").replace("^I", " ")
    text = re.sub(r"\\[AaCcFfHhQqTtWw][^;]*;", "", text)
    text = re.sub(r"\\[LlOoKk]", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def match_text(
    center: tuple[float, float],
    texts: Sequence[dict],
    component_info: dict,
    distance_weight: float,
    keyword_bonus: float,
) -> tuple[str, float, float]:
    """回傳最佳文字、分數及距離；距離分數為 weight/(1+distance)。"""
    if not texts:
        return "", 0.0, math.inf
    keywords = [component_info.get("name", "")]
    keywords += list(component_info.get("english", [])) + list(component_info.get("chinese", []))
    keywords = [_normalized(keyword) for keyword in keywords if str(keyword).strip()]

    best_text, best_score, best_distance = "", -math.inf, math.inf
    for text in texts:
        distance = math.dist(center, text["position"])
        normalized_text = _normalized(text["text"])
        hits = sum(1 for keyword in keywords if keyword and keyword in normalized_text)
        score = distance_weight / (1.0 + distance) + keyword_bonus * hits
        if score > best_score or (math.isclose(score, best_score) and distance < best_distance):
            best_text, best_score, best_distance = text["text"], score, distance
    return best_text, best_score, best_distance


def build_bom(
    df: pd.DataFrame,
    tolerance: float = 0.5,
    distance_weight: float = 100.0,
    keyword_bonus: float = 100.0,
    configured_colors_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometries = [geometry for index, row in df.iterrows() if (geometry := row_to_geometry(int(index), row))]
    if configured_colors_only:
        geometries = [geometry for geometry in geometries if geometry.color in COLOR_COMPONENT_MAP]
    components = cluster_geometries(geometries, tolerance)
    texts = extract_texts(df)

    details = []
    color_sequence: dict[str, int] = {}
    for component in components:
        color = component[0].color
        color_sequence[color] = color_sequence.get(color, 0) + 1
        info = COLOR_COMPONENT_MAP.get(color, {"name": color, "english": [], "chinese": []})
        center = component_center(component)
        matched_text, score, distance = match_text(center, texts, info, distance_weight, keyword_bonus)
        details.append({
            "顏色": color,
            "元件名": info["name"],
            "元件編號": color_sequence[color],
            "規格介紹": matched_text,
            "中心 X": center[0],
            "中心 Y": center[1],
            "配對分數": score,
            "文字距離": distance,
            "圖元數": len(component),
            "圖元列號": ",".join(str(geometry.row_index) for geometry in component),
        })
    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        return pd.DataFrame(columns=["元件名", "規格介紹", "數量"]), detail_df

    # 相同元件名與規格才合併；不同規格各自成一筆 BOM。
    bom = (
        detail_df.groupby(["元件名", "規格介紹"], dropna=False, sort=False)
        .size().reset_index(name="數量")
    )
    return bom, detail_df


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="呼叫 dxf_reader.read_dwg() 分析 DXF/DWG 並產生 BOM")
    parser.add_argument("input", type=Path, help="輸入 DXF/DWG；亦相容既有向量 CSV")
    parser.add_argument("-o", "--output", type=Path, help="BOM CSV；預設為 <input>_bom.csv")
    parser.add_argument("--details-output", type=Path, help="每個聚類元件及配對分數的明細 CSV")
    parser.add_argument("--tolerance", type=float, default=0.5, help="圖元接合/近接容許距離，預設 0.5")
    parser.add_argument("--distance-weight", type=float, default=100.0, help="距離分數權重")
    parser.add_argument("--keyword-bonus", type=float, default=100.0, help="每個名稱關鍵字命中的加分")
    parser.add_argument("--all-colors", action="store_true", help="也處理 COLOR_COMPONENT_MAP 未定義的顏色")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance < 0:
        raise ValueError("--tolerance 不可小於 0")
    output = args.output or args.input.with_name(f"{args.input.stem}_bom.csv")
    details_output = args.details_output or output.with_name(f"{output.stem}_details.csv")
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到輸入檔案：{args.input}")
    df = load_vector_data(args.input)
    bom, details = build_bom(
        df,
        tolerance=args.tolerance,
        distance_weight=args.distance_weight,
        keyword_bonus=args.keyword_bonus,
        configured_colors_only=not args.all_colors,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    details_output.parent.mkdir(parents=True, exist_ok=True)
    bom.to_csv(output, index=False, encoding="utf-8-sig")
    details.to_csv(details_output, index=False, encoding="utf-8-sig")
    print(f"BOM：{output}（{len(bom)} 筆）")
    print(f"元件明細：{details_output}（{len(details)} 個元件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
