"""將原始向量圖與分析結果疊合成可互動的除錯圖。

本模組只讀取向量表、群組明細與 BOM，不參與任何工程判定；
核心套件不匯入本模組，因此沒有 Plotly 也能執行分析。
"""

import html
import math
from typing import Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ezdxf import colors as aci_colors
from .config import TUBE_COLOR
from .geometry import row_to_geometry
from .models import Geometry
from .text import extract_texts

HOVER_PAD_RATIO = 0.004
# 圖面比例：讓較大的元件（第 90 百分位）約佔 COMPONENT_TARGET_PX，再受畫布寬度上下限約束。
COMPONENT_TARGET_PX = 32.0
PLOT_WIDTH_RANGE = (1200.0, 6000.0)
MIN_PLOT_HEIGHT = 420
PLOT_MARGIN = {"l": 20, "r": 20, "t": 100, "b": 20}
# 圖面文字相對於 DXF 字高的縮放；小於 1 可避免標註互相重疊或蓋住元件。
TEXT_HEIGHT_SCALE = 0.65
TEXT_FONT_RANGE = (7.0, 18.0)
MAX_FRAME_HEIGHT = 860
LENGTH_NAMES = {"Tube", "Coil Tube", "Hose", "軟管"}


def _color_rgb(color: str) -> tuple[int, int, int]:
    """把 dxf_reader 的「顏色_N」標籤還原成 RGB；白底上將黑白色一律畫成黑色。"""
    try:
        value = int(str(color).rsplit("_", 1)[-1])
    except ValueError:
        return (0, 0, 0)
    if value in (0, 7) or value == 256:
        return (0, 0, 0)
    if value > 256:
        return (value >> 16 & 255, value >> 8 & 255, value & 255)
    rgb = aci_colors.aci2rgb(value)
    return (rgb.r, rgb.g, rgb.b)


def _css(rgb: tuple[int, int, int], alpha: float = 1.0) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _format_number(value: float) -> str:
    """將數量顯示為簡潔的整數或浮點數。"""
    return str(int(value)) if math.isclose(value, round(value)) else f"{value:g}"


def _hover_polygon(geometries: list[Geometry], pad: float) -> np.ndarray:
    """回傳包住整個圖元組的感應區；單一線段使用沿線方向的細長矩形。"""
    if len(geometries) == 1 and len(geometries[0].points) == 2:
        start, end = geometries[0].points
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length > 0:
            along = delta / length * pad
            normal = np.asarray([-along[1], along[0]])
            return np.asarray([
                start - along + normal, end + along + normal,
                end + along - normal, start - along - normal,
                start - along + normal,
            ])
    points = np.vstack([geometry.points for geometry in geometries])
    min_x, min_y = points.min(axis=0) - pad
    max_x, max_y = points.max(axis=0) + pad
    return np.asarray([
        (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y),
    ])


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))) / 2


def _hover_text(
    row: pd.Series,
    bom_quantity: dict[tuple[str, str, str], object],
    segment=None,
) -> str:
    """顯示元件完整分析資訊，並標示所屬連通塊。"""
    name, size, material = str(row["品名"]), str(row["尺寸"]), str(row["材質"])
    quantity = float(row["原始數量"])
    lines = [f"<b>G{row['群組編號']}：{name}</b>"]
    if segment is not None:
        lines.append(f"所屬連通塊：{segment.code}")
    else:
        lines.append("所屬連通塊：尺寸邊界")
    if name == "IGNORE":
        lines.append("不列入 BOM")
    else:
        lines.append(f"尺寸：{size or '（未判定）'}")
        lines.append(f"材質：{material or '（未判定）'}")
        if name in LENGTH_NAMES:
            lines.append(f"本段長度：{_format_number(quantity)}M")
        total = bom_quantity.get((name, size, material))
        lines.append(f"BOM 合計：{total}" if total is not None else "BOM 合計：未列入")
    if row["配對文字"]:
        lines.append(f"配對文字：{row['配對文字']}")
    lines.append(f"圖元顏色：{row['顏色']}｜圖元數：{row['圖元數']}")
    if row["備註"]:
        lines.append(f"備註：{row['備註']}")
    return "<br>".join(lines)


def _source_rows(value: object) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def build_debug_figure(
    df: pd.DataFrame,
    details: pd.DataFrame,
    bom: pd.DataFrame,
    segments: Sequence | None = None,
) -> go.Figure:
    """畫出原圖，並以連通塊顏色疊加可懸停的分析結果感應區。"""
    geometries = {
        int(index): geometry
        for index, row in df.iterrows()
        if (geometry := row_to_geometry(int(index), row)) and len(geometry.points) > 1
    }
    figure = go.Figure()
    if not geometries:
        return figure

    all_points = np.vstack([geometry.points for geometry in geometries.values()])
    extent = all_points.max(axis=0) - all_points.min(axis=0)
    pad = max(float(extent.max()) * HOVER_PAD_RATIO, 1e-6)

    # 原圖：同色線條合併成單一 trace（以 None 斷開），填充線另外塗滿。
    outlines: dict[str, list[np.ndarray]] = {}
    hatches: dict[str, list[np.ndarray]] = {}
    for geometry in geometries.values():
        target = hatches if geometry.kind == "填充線" else outlines
        target.setdefault(geometry.color, []).append(geometry.points)
    for filled, by_color in ((False, outlines), (True, hatches)):
        for color, paths in by_color.items():
            xs: list[float | None] = []
            ys: list[float | None] = []
            for path in paths:
                xs.extend(path[:, 0].tolist() + [None])
                ys.extend(path[:, 1].tolist() + [None])
            rgb = _color_rgb(color)
            figure.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", line={"color": _css(rgb), "width": 1},
                fill="toself" if filled else None, fillcolor=_css(rgb),
                hoverinfo="skip", showlegend=False,
            ))

    regions = []
    for _, row in details.fillna("").iterrows():
        items = [geometries[index] for index in _source_rows(row["來源列號"]) if index in geometries]
        if not items:
            continue
        polygon = _hover_polygon(items, pad)
        regions.append((_polygon_area(polygon), polygon, row))

    # Gradio 的 gr.Plot 會強制把圖縮進容器，元件會擠成一團；
    # 這裡改依圖面內容決定畫布像素大小，由外層 iframe 提供捲軸。
    component_spans = [
        float(np.ptp(polygon, axis=0).max()) - 2 * pad
        for _, polygon, row in regions
        if row["品名"] != "IGNORE" and row["顏色"] != TUBE_COLOR
    ]
    drawing_width = max(float(extent[0]), 1e-6)
    scale = PLOT_WIDTH_RANGE[0] / drawing_width
    if component_spans:
        reference = max(float(np.percentile(component_spans, 90)), 1e-6)
        scale = max(scale, COMPONENT_TARGET_PX / reference)
    scale = min(scale, PLOT_WIDTH_RANGE[1] / drawing_width)
    texts = extract_texts(df)
    # 文字常標在線條外側，一併納入顯示範圍以免被裁掉。
    bounds = np.vstack([all_points] + [np.asarray([text["position"]]) for text in texts])
    low, high = bounds.min(axis=0) - 8 * pad, bounds.max(axis=0) + 8 * pad
    plot_width = int((high[0] - low[0]) * scale) + PLOT_MARGIN["l"] + PLOT_MARGIN["r"]
    plot_height = max(
        MIN_PLOT_HEIGHT,
        int((high[1] - low[1]) * scale) + PLOT_MARGIN["t"] + PLOT_MARGIN["b"],
    )

    if texts:
        heights = pd.to_numeric(df["高度"], errors="coerce") if "高度" in df.columns else None
        sizes = []
        for text in texts:
            height = heights.get(text["index"]) if heights is not None else None
            pixels = height * scale * TEXT_HEIGHT_SCALE if height is not None and pd.notna(height) else TEXT_FONT_RANGE[0]
            sizes.append(min(TEXT_FONT_RANGE[1], max(TEXT_FONT_RANGE[0], pixels)))
        figure.add_trace(go.Scatter(
            x=[text["position"][0] for text in texts],
            y=[text["position"][1] for text in texts],
            text=[text["text"] for text in texts],
            mode="text", textposition="top right",
            textfont={"size": sizes, "color": "#222"},
            hoverinfo="skip", showlegend=False,
        ))

    bom_quantity = {
        (str(row["品名"]), str(row["尺寸"]), str(row["材質"])): row["數量"]
        for _, row in bom.iterrows()
    }
    segment_by_group = {
        group_id: segment
        for segment in (segments or [])
        for group_id in segment.group_ids
    }
    # 重疊時後畫者優先感應：依序為 IGNORE 外框、管線、元件（面積大的先畫），
    # 讓小元件不會被面板外框或穿過它的管線蓋住。
    def draw_order(item: tuple[float, np.ndarray, pd.Series]) -> tuple[int, float]:
        area, _, row = item
        layer = 0 if row["品名"] == "IGNORE" else 1 if row["顏色"] == TUBE_COLOR else 2
        return layer, -area

    seen_names: set[str] = set()
    for _, polygon, row in sorted(regions, key=draw_order):
        name = str(row["品名"])
        ignored = name == "IGNORE"
        segment = segment_by_group.get(int(row["群組編號"]))
        if segment is not None:
            rgb = tuple(int(segment.color[index:index + 2], 16) for index in (1, 3, 5))
            legend_name = segment.code
        else:
            rgb = (120, 120, 120) if ignored else _color_rgb(row["顏色"])
            legend_name = "尺寸邊界"
        figure.add_trace(go.Scatter(
            x=polygon[:, 0], y=polygon[:, 1], mode="lines",
            line={"color": _css(rgb, 0.55), "width": 1, "dash": "dot"},
            fill="toself", fillcolor=_css(rgb, 0.02 if ignored else 0.10),
            hoveron="fills", hoverinfo="text", text=_hover_text(row, bom_quantity, segment),
            name=legend_name,
            legendgroup=segment.code if segment is not None else legend_name,
            showlegend=legend_name not in seen_names,
        ))
        seen_names.add(legend_name)

    figure.update_layout(
        width=plot_width, height=plot_height, margin=PLOT_MARGIN,
        plot_bgcolor="white", paper_bgcolor="white", dragmode="pan",
        hoverlabel={"align": "left", "bgcolor": "white", "font": {"size": 13}},
        legend={
            "title": {"text": "連通塊"}, "groupclick": "togglegroup",
            "orientation": "h", "x": 0, "xanchor": "left", "y": 1, "yanchor": "bottom",
        },
        xaxis={"visible": False, "range": [float(low[0]), float(high[0])]},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def figure_to_html(figure: go.Figure) -> str:
    """把檢視圖包成自帶 plotly.js 的 iframe，可離線使用並支援滾輪縮放。"""
    document = figure.to_html(
        full_html=True, include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": False},
    )
    frame_height = min(int(figure.layout.height or MIN_PLOT_HEIGHT) + 30, MAX_FRAME_HEIGHT)
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        f'style="width:100%;height:{frame_height}px;border:1px solid #ddd;background:white;"></iframe>'
    )
