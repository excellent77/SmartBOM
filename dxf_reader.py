import os
import subprocess
import shutil
import tempfile
from collections import defaultdict

import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from ezdxf import bbox, colors



PNG_WIDTH = 7680
PNG_HEIGHT = 4320
PNG_DPI = 480
VIEWPORT_PADDING_RATIO = 0.02
QUANTITY_TAGS = {"QTY", "COUNT", "QUANTITY", "NUM", "計數", "數量"}
DRAWABLE_BOUND_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "HATCH"}



def extract_dxf_data(dxf_path):
    """使用 ezdxf 讀取 DXF 資料"""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    data = []
    
    # 萃取圖塊與屬性
    for block in msp.query('INSERT'):
        block_info = {
            "Type": "Block",
            "Name": block.dxf.name,
            "X": round(block.dxf.insert.x, 2),
            "Y": round(block.dxf.insert.y, 2)
        }
        for attrib in block.attribs:
            block_info[attrib.dxf.tag] = attrib.dxf.text
        data.append(block_info)
        
    return pd.DataFrame(data)


def _dxf_attr(entity, name, default=None):
    try:
        return getattr(entity.dxf, name)
    except AttributeError:
        return default


def _layer(doc, entity):
    if doc is None:
        return None
    layer_name = _dxf_attr(entity, "layer")
    if not layer_name or layer_name not in doc.layers:
        return None
    try:
        return doc.layers.get(layer_name)
    except Exception:
        return None


def _entity_color(entity, doc=None) -> str:
    """Return a readable color label, resolving BYLAYER when possible."""
    color = _dxf_attr(entity, "color")

    if color is not None and color != 256:
        return f"顏色_{color}"

    layer = _layer(doc, entity)
    if layer is not None:
        layer_color = _dxf_attr(layer, "color")
        if layer_color is not None and layer_color != 256:
            return f"顏色_{layer_color}"

    true_color = _dxf_attr(entity, "true_color")
    if true_color is not None:
        return f"顏色_{true_color}"
    return "顏色_0"


def _normalize_point(point):
    try:
        x, y = float(point[0]), float(point[1])
        return x, y
    except Exception:
        return None, None


def _true_color_to_rgb(true_color):
    if true_color is None:
        return None
    value = int(true_color)
    return (value >> 16 & 255, value >> 8 & 255, value & 255)


def _aci_to_plot_rgb(color):
    color = abs(int(color))
    if color == 7:
        return (0, 0, 0)
    rgb = colors.aci2rgb(color)
    return (rgb.r, rgb.g, rgb.b)


def _entity_rgb(entity, doc=None):
    rgb = _true_color_to_rgb(_dxf_attr(entity, "true_color"))
    if rgb is not None:
        return rgb

    color = _dxf_attr(entity, "color")
    if color is not None and color not in (0, 256):
        return _aci_to_plot_rgb(color)

    layer = _layer(doc, entity)
    if layer is not None:
        rgb = _true_color_to_rgb(_dxf_attr(layer, "true_color"))
        if rgb is not None:
            return rgb

        layer_color = _dxf_attr(layer, "color")
        if layer_color is not None and layer_color not in (0, 256):
            return _aci_to_plot_rgb(layer_color)

    return (0, 0, 0)


def _rgb_to_matplotlib(rgb):
    return tuple(channel / 255 for channel in rgb)


def _point_xy(point):
    try:
        return float(point[0]), float(point[1])
    except Exception:
        return None


def _degrees(angle):
    angle = float(angle)
    return np.degrees(angle) if abs(angle) <= 2 * np.pi else angle


def _sample_arc_points(center, radius, start_angle, end_angle, ccw=True):
    center = _point_xy(center)
    if center is None:
        return []

    start = _degrees(start_angle)
    end = _degrees(end_angle)
    if ccw and end < start:
        end += 360
    elif not ccw and end > start:
        end -= 360

    steps = max(8, int(abs(end - start) / 10))
    angles = np.radians(np.linspace(start, end, steps + 1))
    return [
        (center[0] + float(radius) * np.cos(angle), center[1] + float(radius) * np.sin(angle))
        for angle in angles
    ]


def _sample_ellipse_points(edge):
    center = _point_xy(getattr(edge, "center", None))
    major_axis = _point_xy(getattr(edge, "major_axis", None))
    if center is None or major_axis is None:
        return []

    ratio = float(getattr(edge, "ratio", 1))
    start = _degrees(getattr(edge, "start_angle", 0))
    end = _degrees(getattr(edge, "end_angle", 360))
    ccw = getattr(edge, "ccw", True)
    if ccw and end < start:
        end += 360
    elif not ccw and end > start:
        end -= 360

    steps = max(16, int(abs(end - start) / 10))
    angles = np.radians(np.linspace(start, end, steps + 1))
    major = np.array(major_axis, dtype=float)
    minor = np.array([-major[1], major[0]], dtype=float) * ratio
    origin = np.array(center, dtype=float)
    return [tuple(origin + np.cos(angle) * major + np.sin(angle) * minor) for angle in angles]


def _hatch_edge_points(edge):
    edge_type = type(edge).__name__.lower()
    if "line" in edge_type:
        return [
            point
            for point in (_point_xy(getattr(edge, "start", None)), _point_xy(getattr(edge, "end", None)))
            if point is not None
        ]
    if "arc" in edge_type and "ellipse" not in edge_type:
        return _sample_arc_points(
            getattr(edge, "center", None),
            getattr(edge, "radius", 0),
            getattr(edge, "start_angle", 0),
            getattr(edge, "end_angle", 0),
            getattr(edge, "ccw", True),
        )
    if "ellipse" in edge_type:
        return _sample_ellipse_points(edge)
    return []


def _hatch_boundary_paths(entity):
    paths = []
    for boundary_path in getattr(entity, "paths", []):
        points = []
        if hasattr(boundary_path, "vertices"):
            points = [_point_xy(vertex) for vertex in boundary_path.vertices]
            points = [point for point in points if point is not None]
            if points and getattr(boundary_path, "is_closed", False) and points[0] != points[-1]:
                points.append(points[0])
        elif hasattr(boundary_path, "edges"):
            for edge in boundary_path.edges:
                edge_points = _hatch_edge_points(edge)
                if not edge_points:
                    continue
                if points and points[-1] == edge_points[0]:
                    points.extend(edge_points[1:])
                else:
                    points.extend(edge_points)
            if points and points[0] != points[-1]:
                points.append(points[0])
        if len(points) >= 2:
            paths.append(points)
    return paths


def _hatch_bounds(entity):
    points = [point for path in _hatch_boundary_paths(entity) for point in path]
    return _bounds_from_points(points)


def _bounds_from_points(points):
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _circle_bounds(center, radius):
    if center is None or None in center:
        return None
    radius = float(radius)
    return center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius


def _draw_hatch(ax, entity, draw_color):
    paths = _hatch_boundary_paths(entity)
    if not paths:
        return False

    solid_fill = bool(getattr(entity.dxf, "solid_fill", 0))
    for points in paths:
        if solid_fill and len(points) >= 3:
            polygon = patches.Polygon(
                points,
                closed=True,
                facecolor=draw_color,
                edgecolor=draw_color,
                alpha=0.3,
                linewidth=1,
            )
            ax.add_patch(polygon)
        else:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, color=draw_color, linewidth=1)
    return True


def _entity_bounds(entity, entity_type):
    if entity_type == "LINE":
        start = _normalize_point(entity.dxf.start)
        end = _normalize_point(entity.dxf.end)
        return _bounds_from_points([start, end]) if None not in start and None not in end else None

    if entity_type in {"LWPOLYLINE", "POLYLINE"}:
        points = []
        for start, end in _explode_polyline(entity):
            points.extend([start, end])
        return _bounds_from_points(points)

    if entity_type == "CIRCLE":
        return _circle_bounds(_normalize_point(entity.dxf.center), entity.dxf.radius)

    if entity_type == "ARC":
        return _circle_bounds(_normalize_point(entity.dxf.center), entity.dxf.radius)

    if entity_type == "ELLIPSE":
        center = _normalize_point(entity.dxf.center)
        if None in center:
            return None
        semi_major = entity.dxf.major_axis.magnitude
        semi_minor = entity.dxf.minor_axis.magnitude
        return center[0] - semi_major, center[1] - semi_minor, center[0] + semi_major, center[1] + semi_minor

    if entity_type == "HATCH":
        return _hatch_bounds(entity)

    if entity_type in {"TEXT", "MTEXT", "INSERT"}:
        point = _normalize_point(entity.dxf.insert)
        return _bounds_from_points([point]) if None not in point else None

    if entity_type == "POINT":
        point = _normalize_point(entity.dxf.location)
        return _bounds_from_points([point]) if None not in point else None

    return None


def _bounds_intersect(first, second):
    if first is None or second is None:
        return True
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _layout_viewport_bounds(doc, padding_ratio=0.02):
    """
    Return modelspace bounds shown by the active paper-space viewport.

    DXF VIEWPORT stores paper-space size and model-space view center/height.
    The paper viewport with status 1 is the overall paper view; model viewports
    have status > 1, so only those should be used for clipping model entities.
    """
    candidates = []
    for layout in doc.layouts:
        if layout.name.lower() == "model":
            continue
        for viewport in layout.query("VIEWPORT"):
            status = int(getattr(viewport.dxf, "status", 0) or 0)
            if status <= 1:
                continue
            try:
                paper_width = float(viewport.dxf.width)
                paper_height = float(viewport.dxf.height)
                view_height = float(viewport.dxf.view_height)
                view_center = viewport.dxf.view_center_point
            except Exception:
                continue
            if paper_width <= 0 or paper_height <= 0 or view_height <= 0:
                continue

            view_width = view_height * paper_width / paper_height
            min_x = float(view_center[0]) - view_width / 2
            max_x = float(view_center[0]) + view_width / 2
            min_y = float(view_center[1]) - view_height / 2
            max_y = float(view_center[1]) + view_height / 2
            padding = max(view_width, view_height) * padding_ratio
            candidates.append((
                min_x - padding,
                min_y - padding,
                max_x + padding,
                max_y + padding,
            ))

    if not candidates:
        return None

    return (
        min(bounds[0] for bounds in candidates),
        min(bounds[1] for bounds in candidates),
        max(bounds[2] for bounds in candidates),
        max(bounds[3] for bounds in candidates),
    )


def _iter_modelspace_entities(doc, use_layout_viewport=True):
    clip_bounds = _layout_viewport_bounds(doc) if use_layout_viewport else None
    for entity in doc.modelspace():
        entity_type = entity.dxftype()
        if clip_bounds is None or _bounds_intersect(_entity_bounds(entity, entity_type), clip_bounds):
            yield entity, entity_type


def _extend_entity_bounds(entity, entity_type, all_x, all_y):
    if entity_type not in DRAWABLE_BOUND_TYPES:
        return
    bounds = _entity_bounds(entity, entity_type)
    if bounds is None:
        return
    min_x, min_y, max_x, max_y = bounds
    all_x.extend([min_x, max_x])
    all_y.extend([min_y, max_y])


def _extract_count_from_attribs(attribs):
    count = 1
    numeric_candidates = []
    for attrib in attribs:
        tag = getattr(attrib.dxf, "tag", "").upper() if hasattr(attrib, "dxf") else ""
        text = getattr(attrib.dxf, "text", "") if hasattr(attrib, "dxf") else ""
        if text is None:
            continue
        s = str(text).strip()
        if not s:
            continue
        if s.isdigit():
            numeric = int(s)
            numeric_candidates.append((tag, numeric))
        else:
            try:
                numeric = int(float(s))
                numeric_candidates.append((tag, numeric))
            except Exception:
                continue
    for tag, numeric in numeric_candidates:
        if tag in QUANTITY_TAGS:
            return numeric
    if numeric_candidates:
        return numeric_candidates[-1][1]
    return count


def _add_row(rows: list, name: str, color: str, values: dict, count: int = 1):
    row = {
        "名稱": name,
        "出圖型式": color,
        "計數": count,
    }
    row.update(values)
    rows.append(row)


def _text_bounds_center(entity):
    """Return the visual bounding-box center for TEXT/MTEXT when available."""
    try:
        bounds = bbox.extents([entity])
        if bounds.has_data:
            center = bounds.center
            return float(center.x), float(center.y)
    except Exception:
        pass
    return None


def _explode_polyline(entity):
    if hasattr(entity, "get_points"):
        points = [tuple(pt[:2]) for pt in entity.get_points()]
    else:
        points = []
        for vertex in entity.vertices():
            try:
                location = vertex.dxf.location
                points.append((float(location.x), float(location.y)))
            except Exception:
                continue
    if len(points) < 2:
        return []
    segments = []
    for index in range(len(points) - 1):
        segments.append((points[index], points[index + 1]))
    if getattr(entity, "is_closed", False) or getattr(entity, "closed", False):
        segments.append((points[-1], points[0]))
    return segments

def path_to_dxf(file_path: str):
    file_path = os.path.abspath(file_path)
    base, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".dxf":
        return ezdxf.readfile(file_path)

    if not shutil.which("ODAFileConverter"):
        raise EnvironmentError("找不到 ODAFileConverter，無法將 DWG 轉為 DXF。請安裝並加入 PATH。")

    input_dir = os.path.dirname(file_path) or "."
    with tempfile.TemporaryDirectory(prefix="dxf_conv_") as tmp_out:
        xvfb_cmd = ["xvfb-run", "-a"] if shutil.which("xvfb-run") else []
        cmd = xvfb_cmd + [
            "ODAFileConverter",
            input_dir,
            tmp_out,
            "ACAD2018",
            "DXF",
            "0",
            "1",
            os.path.basename(file_path),
        ]

        print(f"啟動轉檔程序...\n指令: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        expected = os.path.join(tmp_out, os.path.basename(base) + ".dxf")
        if not os.path.exists(expected):
            candidates = [
                os.path.join(tmp_out, name)
                for name in os.listdir(tmp_out)
                if name.lower().endswith(".dxf")
            ]
            if not candidates:
                raise FileNotFoundError(f"轉換失敗或找不到產生的 DXF 檔案。ODA stderr:\n{result.stderr}")
            expected = candidates[0]

        return ezdxf.readfile(expected)


def _rows_for_entity(entity, entity_type, color):
    rows = []

    if entity_type == "LINE":
        start = _normalize_point(entity.dxf.start)
        end = _normalize_point(entity.dxf.end)
        _add_row(rows, "線", color, {
            "起點 X": start[0],
            "起點 Y": start[1],
            "終點 X": end[0],
            "終點 Y": end[1],
        })

    elif entity_type in {"LWPOLYLINE", "POLYLINE"}:
        name = "聚合線" if getattr(entity, "is_closed", False) or getattr(entity, "closed", False) else "線"
        for index, (start, end) in enumerate(_explode_polyline(entity)):
            _add_row(rows, name, color, {
                "起點 X": float(start[0]),
                "起點 Y": float(start[1]),
                "終點 X": float(end[0]),
                "終點 Y": float(end[1]),
            }, count=1 if index == 0 else 0)

    elif entity_type == "CIRCLE":
        center = _normalize_point(entity.dxf.center)
        _add_row(rows, "圓", color, {
            "中心點 X": center[0],
            "中心點 Y": center[1],
            "半徑": float(entity.dxf.radius),
        })

    elif entity_type == "ARC":
        center = _normalize_point(entity.dxf.center)
        try:
            start_point = _normalize_point(entity.start_point)
            end_point = _normalize_point(entity.end_point)
        except Exception:
            start_point = (None, None)
            end_point = (None, None)
        _add_row(rows, "弧", color, {
            "中心點 X": center[0],
            "中心點 Y": center[1],
            "起點 X": start_point[0],
            "起點 Y": start_point[1],
            "終點 X": end_point[0],
            "終點 Y": end_point[1],
            "半徑": float(entity.dxf.radius),
        })

    elif entity_type == "ELLIPSE":
        center = _normalize_point(entity.dxf.center)
        _add_row(rows, "橢圓", color, {
            "中心點 X": center[0],
            "中心點 Y": center[1],
        })

    elif entity_type == "TEXT":
        insert = _normalize_point(entity.dxf.insert)
        values = {
            "位置 X": insert[0],
            "位置 Y": insert[1],
            "值": entity.dxf.text,
            "高度": float(getattr(entity.dxf, "height", 0)),
            "寬度係數": float(getattr(entity.dxf, "width", 1)),
            "旋轉": float(getattr(entity.dxf, "rotation", 0)),
        }
        text_center = _text_bounds_center(entity)
        if text_center is not None:
            values.update({"文字中心 X": text_center[0], "文字中心 Y": text_center[1]})
        _add_row(rows, "文字", color, values)

    elif entity_type == "MTEXT":
        insert = _normalize_point(entity.dxf.insert)
        values = {
            "位置 X": insert[0],
            "位置 Y": insert[1],
            "值": entity.text,
            "高度": float(getattr(entity.dxf, "char_height", 0)),
            "寬度": float(getattr(entity.dxf, "width", 0)),
            "旋轉": float(getattr(entity.dxf, "rotation", 0)),
        }
        text_center = _text_bounds_center(entity)
        if text_center is not None:
            values.update({"文字中心 X": text_center[0], "文字中心 Y": text_center[1]})
        _add_row(rows, "多行文字", color, values)

    elif entity_type == "INSERT":
        insert = _normalize_point(entity.dxf.insert)
        row = {
            "名稱": entity.dxf.name or "INSERT",
            "出圖型式": color,
            "計數": _extract_count_from_attribs(entity.attribs),
            "X": insert[0],
            "Y": insert[1],
        }
        for attrib in entity.attribs:
            row[attrib.dxf.tag] = attrib.dxf.text
        rows.append(row)

    elif entity_type == "HATCH":
        values = {}
        bounds = _hatch_bounds(entity)
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            values.update({
                "起點 X": min_x,
                "起點 Y": min_y,
                "終點 X": max_x,
                "終點 Y": max_y,
            })
        _add_row(rows, "填充線", color, values)

    return rows


def _draw_entity(ax, entity, entity_type, draw_color):
    if entity_type == "LINE":
        start = _normalize_point(entity.dxf.start)
        end = _normalize_point(entity.dxf.end)
        if None not in start and None not in end:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=draw_color, linewidth=1)

    elif entity_type in {"LWPOLYLINE", "POLYLINE"}:
        for start, end in _explode_polyline(entity):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=draw_color, linewidth=1)

    elif entity_type == "CIRCLE":
        center = _normalize_point(entity.dxf.center)
        if None not in center:
            circle = patches.Circle(center, float(entity.dxf.radius), fill=False, edgecolor=draw_color, linewidth=1)
            ax.add_patch(circle)

    elif entity_type == "ARC":
        center = _normalize_point(entity.dxf.center)
        if None not in center:
            radius = float(entity.dxf.radius)
            start_angle = float(getattr(entity.dxf, "start_angle", 0))
            end_angle = float(getattr(entity.dxf, "end_angle", 360))
            arc = patches.Arc(
                center,
                2 * radius,
                2 * radius,
                angle=0,
                theta1=start_angle,
                theta2=end_angle,
                color=draw_color,
                linewidth=1,
            )
            ax.add_patch(arc)

    elif entity_type == "ELLIPSE":
        center = _normalize_point(entity.dxf.center)
        if None not in center:
            ellipse = patches.Ellipse(
                center,
                2 * entity.dxf.major_axis.magnitude,
                2 * entity.dxf.minor_axis.magnitude,
                angle=np.degrees(entity.dxf.rotation),
                fill=False,
                edgecolor=draw_color,
                linewidth=1,
            )
            ax.add_patch(ellipse)

    elif entity_type == "HATCH":
        _draw_hatch(ax, entity, draw_color)


def _axis_limits_from_bounds(xs, ys):
    if not xs or not ys:
        return None
    margin = 0.1 * (max(xs) - min(xs) + max(ys) - min(ys)) / 2
    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )


def read_dwg(file_path: str, use_layout_viewport: bool = True) -> pd.DataFrame:
    """Read a DWG or DXF path. If DWG, attempt to convert to DXF using ODAFileConverter.

    Returns a DataFrame compatible with the rest of the codebase.
    """

    doc = path_to_dxf(file_path)
    rows = []

    for entity, entity_type in _iter_modelspace_entities(doc, use_layout_viewport):
        color = _entity_color(entity, doc)
        rows.extend(_rows_for_entity(entity, entity_type, color))

    return pd.DataFrame(rows)


def export_dxf_by_color_to_png(dxf_path: str, output_dir: str = None, use_layout_viewport: bool = True):
    """
    讀入DXF檔案，根據modelspace中entity的顏色資訊分組，
    每個顏色生成一個PNG檔案。
    
    Args:
        dxf_path: DXF或DWG檔案路徑
        output_dir: 輸出目錄，若為None則在同目錄下建立 colors_png 資料夾
    
    Returns:
        dict: {顏色名稱: PNG檔案路徑}
    """
    file_path = os.path.abspath(dxf_path)
    doc = path_to_dxf(dxf_path)
    
    # 根據顏色分組entity
    entities_by_color = defaultdict(list)
    global_x, global_y = [], []
    
    for entity, entity_type in _iter_modelspace_entities(doc, use_layout_viewport):
        color = _entity_color(entity, doc)
        entities_by_color[color].append((entity, entity_type))
        try:
            _extend_entity_bounds(entity, entity_type, global_x, global_y)
        except Exception:
            pass

    global_axis_limits = _axis_limits_from_bounds(global_x, global_y)
    
    # 建立輸出目錄
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_path), "colors_png")
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    figure_size = (PNG_WIDTH / PNG_DPI, PNG_HEIGHT / PNG_DPI)
    
    # 為每個顏色生成PNG
    for color, entities in entities_by_color.items():
        fig, ax = plt.subplots(figsize=figure_size, dpi=PNG_DPI)
        
        for entity, entity_type in entities:
            try:
                draw_color = _rgb_to_matplotlib(_entity_rgb(entity, doc))
                _draw_entity(ax, entity, entity_type, draw_color)
            
            except Exception as e:
                print(f"警告: 無法繪製 {entity_type} entity: {e}")
                continue
        
        # 所有顏色圖使用同一組座標軸範圍，避免每張PNG自動縮放。
        if global_axis_limits:
            min_x, max_x, min_y, max_y = global_axis_limits
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_autoscale_on(False)
        
        ax.set_aspect('equal')
        ax.set_title(f"DXF - {color[3:]}")
        ax.grid(True, alpha=0.3)
        
        # 保存PNG
        file_name = f"{color}.png"
        output_path = os.path.join(output_dir, file_name)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.04)
        fig.savefig(output_path, dpi=PNG_DPI)
        plt.close(fig)
        
        output_files[color] = output_path
        print(f"已生成: {output_path}")
    
    return output_files



if __name__ == "__main__":
    #path = "/home/f11167/SmartBOM/data/16_單線圖_KOXDLP2000.dxf"
    import os
    for file in os.listdir("/home/f11167/SmartBOM/data_new"):
        try:
            code = int(file.split("_")[0])
        except Exception:
            code = 10000
        if file.lower().endswith((".dwg")):
            path = os.path.join("/home/f11167/SmartBOM/data_new", file)
            export_dxf_by_color_to_png(path, os.path.join("/home/f11167/SmartBOM/", file))
            df = read_dwg(path)
            df.to_csv("/home/f11167/SmartBOM/test.csv", index=False, encoding='utf-8-sig')
