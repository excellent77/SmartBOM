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
from .config import COMPONENT_TYPE_TUBE, TUBE_COLOR
from .geometry import row_to_geometry
from .models import Geometry
from .text import extract_texts

HOVER_PAD_RATIO = 0.004
# 圖面比例：讓較大的元件（第 90 百分位）約佔 COMPONENT_TARGET_PX，再受畫布寬度上下限約束。
COMPONENT_TARGET_PX = 32.0
PLOT_WIDTH_RANGE = (1200.0, 6000.0)
MIN_PLOT_HEIGHT = 420
PLOT_MARGIN = {"l": 20, "r": 20, "t": 20, "b": 20}
# 圖面文字相對於 DXF 字高的縮放；小於 1 可避免標註互相重疊或蓋住元件。
TEXT_HEIGHT_SCALE = 0.65
TEXT_FONT_RANGE = (7.0, 18.0)
MAX_FRAME_HEIGHT = 860
LENGTH_NAMES = {"Tube", "Coil Tube", "Hose", "軟管"}
# 配對文字與判定欄位有差異的元件，以醒目紅框高亮。
MISMATCH_RGB = (220, 38, 38)
# 圖內「顯示差異」開關依 trace.meta 切換：開啟時顯示 MISMATCH_META，
# 關閉時改顯示 MISMATCH_OFF_META（無警示的一般標籤）。
MISMATCH_META = "mismatch"
MISMATCH_OFF_META = "mismatch-off"
# 固定在 iframe 左上角的開關，捲動畫布時仍可操作；切換不重繪圖面，保留縮放位置。
MISMATCH_TOGGLE_SCRIPT = """
var gd = document.getElementById('{plot_id}');
var box = document.createElement('label');
box.style.cssText = 'position:fixed;top:8px;left:8px;z-index:1000;background:#fff;'
  + 'border:1px solid #DC2626;border-radius:6px;padding:4px 10px;cursor:pointer;'
  + 'font:13px sans-serif;color:#DC2626;box-shadow:0 1px 4px rgba(0,0,0,.15);';
box.innerHTML = '<input type="checkbox" checked style="vertical-align:middle"> 顯示差異';
document.body.appendChild(box);
box.querySelector('input').addEventListener('change', function (event) {
  var on = event.target.checked, shown = [], hidden = [];
  gd.data.forEach(function (trace, index) {
    if (trace.meta === '%s') shown.push(index);
    else if (trace.meta === '%s') hidden.push(index);
  });
  if (shown.length) Plotly.restyle(gd, {visible: on}, shown);
  if (hidden.length) Plotly.restyle(gd, {visible: !on}, hidden);
});
""" % (MISMATCH_META, MISMATCH_OFF_META)


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
    issues: Sequence[str] = (),
) -> str:
    """顯示元件完整分析資訊，並標示所屬連通塊與差異。"""
    name, size, material = str(row["品名"]), str(row["尺寸"]), str(row["材質"])
    quantity = float(row["原始數量"])
    lines = [f"<b>G{row['群組編號']}：{name}</b>"]
    for issue in issues:
        lines.append(f"<span style='color:{_css(MISMATCH_RGB)}'><b>⚠ {html.escape(issue)}</b></span>")
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
    """畫出原圖與連通塊標籤，並以不可見感應區提供元件 Hover。"""
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
        if row["品名"] != "IGNORE"
        and row.get("元件類型", "") != COMPONENT_TYPE_TUBE
        and not ("元件類型" not in row and row["顏色"] == TUBE_COLOR)
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
        row_is_tube = (
            row.get("元件類型", "") == COMPONENT_TYPE_TUBE
            or ("元件類型" not in row and row["顏色"] == TUBE_COLOR)
        )
        layer = 0 if row["品名"] == "IGNORE" else 1 if row_is_tube else 2
        return layer, -area

    label_candidates: dict[int, list[np.ndarray]] = {}
    mismatch_polygons: list[np.ndarray] = []
    for _, polygon, row in sorted(regions, key=draw_order):
        name = str(row["品名"])
        ignored = name == "IGNORE"
        segment = segment_by_group.get(int(row["群組編號"]))
        if segment is not None:
            label_candidates.setdefault(segment.segment_id, []).append(
                polygon[:-1].mean(axis=0)
            )
            rgb = tuple(int(segment.color[index:index + 2], 16) for index in (1, 3, 5))
        else:
            rgb = (120, 120, 120) if ignored else _color_rgb(row["顏色"])
        issues = segment.mismatches.get(int(row["群組編號"]), []) if segment is not None else []
        if issues:
            mismatch_polygons.append(polygon)
        figure.add_trace(go.Scatter(
            x=polygon[:, 0], y=polygon[:, 1], mode="lines",
            line={"color": _css(rgb, 0.55), "width": 1, "dash": "dot"},
            fill="toself", fillcolor=_css(rgb, 0.02 if ignored else 0.10),
            hoveron="fills", hoverinfo="text",
            text=_hover_text(row, bom_quantity, segment, issues),
            showlegend=False,
        ))

    # 差異高亮獨立成一條 trace（以 meta 標記），讓圖內開關可只切換它；
    # hoverinfo="skip" 使其不攔截下方元件的 Hover。
    if mismatch_polygons:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for polygon in mismatch_polygons:
            xs.extend(polygon[:, 0].tolist() + [None])
            ys.extend(polygon[:, 1].tolist() + [None])
        figure.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", meta=MISMATCH_META,
            line={"color": _css(MISMATCH_RGB), "width": 3},
            fill="toself", fillcolor=_css(MISMATCH_RGB, 0.18),
            hoverinfo="skip", showlegend=False,
        ))

    # 標籤放在最接近該連通塊幾何中心的實際元件上，避免漂浮在空白處。
    # 有差異的連通塊同時準備黑色與紅色「⚠N」兩種標籤，由圖內開關切換。
    labels: dict[str | None, tuple[list[float], list[float], list[str], str]] = {
        None: ([], [], [], "#000000"),
        MISMATCH_OFF_META: ([], [], [], "#000000"),
        MISMATCH_META: ([], [], [], _css(MISMATCH_RGB)),
    }
    for segment in segments or []:
        candidates = label_candidates.get(segment.segment_id, [])
        if not candidates:
            continue
        points = np.vstack(candidates)
        center = points.mean(axis=0)
        position = min(points, key=lambda point: float(np.linalg.norm(point - center)))
        variants = (
            [(MISMATCH_OFF_META, segment.code),
             (MISMATCH_META, f"{segment.code} ⚠{segment.mismatch_count}")]
            if segment.mismatch_count else [(None, segment.code)]
        )
        for meta, text in variants:
            label_x, label_y, label_text, _ = labels[meta]
            label_x.append(float(position[0]))
            label_y.append(float(position[1]))
            label_text.append(f"<b>{text}</b>")
    for meta, (label_x, label_y, label_text, color) in labels.items():
        if not label_text:
            continue
        figure.add_trace(go.Scatter(
            x=label_x, y=label_y, text=label_text, meta=meta,
            visible=meta != MISMATCH_OFF_META,
            mode="text", textposition="top center",
            textfont={"size": 14, "color": color},
            hoverinfo="skip", showlegend=False, cliponaxis=False,
        ))

    figure.update_layout(
        width=plot_width, height=plot_height, margin=PLOT_MARGIN,
        plot_bgcolor="white", paper_bgcolor="white", dragmode="pan",
        hoverlabel={"align": "left", "bgcolor": "white", "font": {"size": 13}},
        showlegend=False,
        xaxis={"visible": False, "range": [float(low[0]), float(high[0])]},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def figure_to_html(figure: go.Figure) -> str:
    """把檢視圖包成自帶 plotly.js 的 iframe，可離線使用並支援滾輪縮放。"""
    has_mismatch = any(trace.meta == MISMATCH_META for trace in figure.data)
    document = figure.to_html(
        full_html=True, include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": False},
        post_script=MISMATCH_TOGGLE_SCRIPT if has_mismatch else None,
    )
    frame_height = min(int(figure.layout.height or MIN_PLOT_HEIGHT) + 30, MAX_FRAME_HEIGHT)
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        f'style="width:100%;height:{frame_height}px;border:1px solid #ddd;background:white;"></iframe>'
    )
