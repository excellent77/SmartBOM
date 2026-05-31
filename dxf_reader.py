import os
import ezdxf
import pandas as pd
import subprocess
import shutil



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


def _entity_color(entity, doc=None) -> str:
    """Return a readable plotter/color label for DXF entities."""
    color = None
    try:
        color = entity.dxf.color
    except AttributeError:
        color = None

    if color is not None and color != 256:
        return f"顏色_{color}"

    # BYLAYER or BYBLOCK: try layer color
    if doc is not None:
        try:
            layer_name = entity.dxf.layer
            if layer_name and layer_name in doc.layers:
                layer = doc.layers.get(layer_name)
                layer_color = layer.dxf.color
                if layer_color is not None and layer_color != 256:
                    return f"顏色_{layer_color}"
        except Exception:
            pass

    try:
        true_color = entity.dxf.true_color
    except AttributeError:
        true_color = None

    if true_color is not None:
        return f"顏色_{true_color}"
    return "顏色_0"


def _normalize_point(point):
    try:
        x, y = float(point[0]), float(point[1])
        return x, y
    except Exception:
        return None, None


def _extract_count_from_attribs(attribs):
    count = 1
    qty_tags = {"QTY", "COUNT", "QUANTITY", "NUM", "計數", "數量"}
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
        if tag in qty_tags:
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


def read_dwg(file_path: str) -> pd.DataFrame:
    """Read a DWG or DXF path. If DWG, attempt to convert to DXF using ODAFileConverter.

    Returns a DataFrame compatible with the rest of the codebase.
    """
    file_path = os.path.abspath(file_path)
    base, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # If already a DXF, read directly
    if ext == ".dxf":
        doc = ezdxf.readfile(file_path)
    else:
        # For DWG input, convert using ODAFileConverter into a temporary directory
        import tempfile

        if not shutil.which("ODAFileConverter"):
            raise EnvironmentError("找不到 ODAFileConverter，無法將 DWG 轉為 DXF。請安裝並加入 PATH。")

        tmp_out = tempfile.mkdtemp(prefix="dxf_conv_")
        input_dir = os.path.dirname(file_path) or "."

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

        # look for converted file with same basename but .dxf
        expected = os.path.join(tmp_out, os.path.basename(base) + ".dxf")
        if not os.path.exists(expected):
            # try to find any dxf in tmp_out
            candidates = [os.path.join(tmp_out, f) for f in os.listdir(tmp_out) if f.lower().endswith('.dxf')]
            if not candidates:
                # include stderr for debugging
                raise FileNotFoundError(f"轉換失敗或找不到產生的 DXF 檔案。ODA stderr:\n{result.stderr}")
            expected = candidates[0]

        try:
            doc = ezdxf.readfile(expected)
        finally:
            # attempt cleanup of temporary files
            try:
                shutil.rmtree(tmp_out)
            except Exception:
                pass
    msp = doc.modelspace()
    rows = []

    for entity in msp:
        entity_type = entity.dxftype()
        color = _entity_color(entity, doc)

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
            segments = _explode_polyline(entity)
            for index, (start, end) in enumerate(segments):
                count = 1 if index == 0 else 0
                _add_row(rows, name, color, {
                    "起點 X": float(start[0]),
                    "起點 Y": float(start[1]),
                    "終點 X": float(end[0]),
                    "終點 Y": float(end[1]),
                }, count=count)

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
            _add_row(rows, "文字", color, {
                "位置 X": insert[0],
                "位置 Y": insert[1],
                "值": entity.dxf.text,
                "旋轉": float(getattr(entity.dxf, "rotation", 0)),
            })

        elif entity_type == "MTEXT":
            insert = _normalize_point(entity.dxf.insert)
            _add_row(rows, "多行文字", color, {
                "位置 X": insert[0],
                "位置 Y": insert[1],
                "值": entity.text,
                "旋轉": float(getattr(entity.dxf, "rotation", 0)),
            })

        elif entity_type == "INSERT":
            insert = _normalize_point(entity.dxf.insert)
            count = _extract_count_from_attribs(entity.attribs)
            row = {
                "名稱": entity.dxf.name or "INSERT",
                "出圖型式": color,
                "計數": count,
                "X": insert[0],
                "Y": insert[1],
            }
            for attrib in entity.attribs:
                row[attrib.dxf.tag] = attrib.dxf.text
            rows.append(row)

        elif entity_type == "HATCH":
            # HATCH can represent filled geometry; count it as a filled line object.
            bbox = entity.extents() if hasattr(entity, "extents") else None
            values = {}
            if bbox is not None:
                values["起點 X"] = float(bbox.extmin.x)
                values["起點 Y"] = float(bbox.extmin.y)
                values["終點 X"] = float(bbox.extmax.x)
                values["終點 Y"] = float(bbox.extmax.y)
            _add_row(rows, "填充線", color, values)

    return pd.DataFrame(rows)