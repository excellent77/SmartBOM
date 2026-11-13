import os
import subprocess
import shutil
import tempfile

import ezdxf
import numpy as np
import pandas as pd
from ezdxf import bbox
from columns import (
    COL_NAME, COL_COLOR, COL_COUNT,
    COL_START_X, COL_START_Y, COL_END_X, COL_END_Y,
    COL_CENTER_X, COL_CENTER_Y,
    COL_POS_X, COL_POS_Y, COL_VALUE, COL_ROTATION, COL_RADIUS,
    COL_HEIGHT, COL_WIDTH, COL_WIDTH_FACTOR, COL_TEXT_CENTER_X, COL_TEXT_CENTER_Y,
    ETYPE_LINE, ETYPE_POLYLINE, ETYPE_CIRCLE, ETYPE_ELLIPSE,
    ETYPE_ARC, ETYPE_TEXT, ETYPE_MTEXT, ETYPE_HATCH,
    round_coord,
)



VIEWPORT_PADDING_RATIO = 0.02
QUANTITY_TAGS = {"QTY", "COUNT", "QUANTITY", "NUM", "計數", "數量"}



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
        COL_NAME: name,
        COL_COLOR: color,
        COL_COUNT: count,
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
        for vertex in entity.vertices:
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
        _add_row(rows, ETYPE_LINE, color, {
            COL_START_X: round_coord(start[0]),
            COL_START_Y: round_coord(start[1]),
            COL_END_X: round_coord(end[0]),
            COL_END_Y: round_coord(end[1]),
        })

    elif entity_type in {"LWPOLYLINE", "POLYLINE"}:
        name = ETYPE_POLYLINE if getattr(entity, "is_closed", False) or getattr(entity, "closed", False) else ETYPE_LINE
        for index, (start, end) in enumerate(_explode_polyline(entity)):
            _add_row(rows, name, color, {
                COL_START_X: round_coord(float(start[0])),
                COL_START_Y: round_coord(float(start[1])),
                COL_END_X: round_coord(float(end[0])),
                COL_END_Y: round_coord(float(end[1])),
            }, count=1 if index == 0 else 0)

    elif entity_type == "CIRCLE":
        center = _normalize_point(entity.dxf.center)
        _add_row(rows, ETYPE_CIRCLE, color, {
            COL_CENTER_X: round_coord(center[0]),
            COL_CENTER_Y: round_coord(center[1]),
            COL_RADIUS: float(entity.dxf.radius),
        })

    elif entity_type == "ARC":
        center = _normalize_point(entity.dxf.center)
        try:
            start_point = _normalize_point(entity.start_point)
            end_point = _normalize_point(entity.end_point)
        except Exception:
            start_point = (None, None)
            end_point = (None, None)
        _add_row(rows, ETYPE_ARC, color, {
            COL_CENTER_X: round_coord(center[0]),
            COL_CENTER_Y: round_coord(center[1]),
            COL_START_X: round_coord(start_point[0]) if start_point[0] is not None else None,
            COL_START_Y: round_coord(start_point[1]) if start_point[1] is not None else None,
            COL_END_X: round_coord(end_point[0]) if end_point[0] is not None else None,
            COL_END_Y: round_coord(end_point[1]) if end_point[1] is not None else None,
            COL_RADIUS: float(entity.dxf.radius),
        })

    elif entity_type == "ELLIPSE":
        center = _normalize_point(entity.dxf.center)
        _add_row(rows, ETYPE_ELLIPSE, color, {
            COL_CENTER_X: round_coord(center[0]),
            COL_CENTER_Y: round_coord(center[1]),
        })

    elif entity_type == "TEXT":
        insert = _normalize_point(entity.dxf.insert)
        values = {
            COL_POS_X: round_coord(insert[0]),
            COL_POS_Y: round_coord(insert[1]),
            COL_VALUE: entity.dxf.text,
            COL_HEIGHT: float(getattr(entity.dxf, "height", 0)),
            COL_WIDTH_FACTOR: float(getattr(entity.dxf, "width", 1)),
            COL_ROTATION: float(getattr(entity.dxf, "rotation", 0)),
        }
        text_center = _text_bounds_center(entity)
        if text_center is not None:
            values.update({COL_TEXT_CENTER_X: round_coord(text_center[0]), COL_TEXT_CENTER_Y: round_coord(text_center[1])})
        _add_row(rows, ETYPE_TEXT, color, values)

    elif entity_type == "MTEXT":
        insert = _normalize_point(entity.dxf.insert)
        values = {
            COL_POS_X: round_coord(insert[0]),
            COL_POS_Y: round_coord(insert[1]),
            COL_VALUE: entity.text,
            COL_HEIGHT: float(getattr(entity.dxf, "char_height", 0)),
            COL_WIDTH: float(getattr(entity.dxf, "width", 0)),
            COL_ROTATION: float(getattr(entity.dxf, "rotation", 0)),
        }
        text_center = _text_bounds_center(entity)
        if text_center is not None:
            values.update({COL_TEXT_CENTER_X: round_coord(text_center[0]), COL_TEXT_CENTER_Y: round_coord(text_center[1])})
        _add_row(rows, ETYPE_MTEXT, color, values)

    elif entity_type == "INSERT":
        insert = _normalize_point(entity.dxf.insert)
        row = {
            COL_NAME: entity.dxf.name or "INSERT",
            COL_COLOR: color,
            COL_COUNT: _extract_count_from_attribs(entity.attribs),
            "X": round_coord(insert[0]),
            "Y": round_coord(insert[1]),
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
                COL_START_X: round_coord(min_x),
                COL_START_Y: round_coord(min_y),
                COL_END_X: round_coord(max_x),
                COL_END_Y: round_coord(max_y),
            })
        _add_row(rows, ETYPE_HATCH, color, values)

    return rows


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
