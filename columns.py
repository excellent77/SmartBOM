"""SmartBOM 欄位名稱常數模組。

集中定義所有 DataFrame 欄位名、Graph 邊屬性鍵名與顯示 / 匯出用欄位名，
避免硬編碼字串散落於各檔案中。
"""

# === DXF 原始 DataFrame 欄位 (中文，保持與 DXF 語義一致) ===
COL_NAME = "名稱"
COL_COLOR = "出圖型式"
COL_COUNT = "計數"
COL_START_X = "起點 X"
COL_START_Y = "起點 Y"
COL_END_X = "終點 X"
COL_END_Y = "終點 Y"
COL_CENTER_X = "中心點 X"
COL_CENTER_Y = "中心點 Y"
COL_POS_X = "位置 X"
COL_POS_Y = "位置 Y"
COL_VALUE = "值"
COL_ROTATION = "旋轉"
COL_RADIUS = "半徑"
COL_HEIGHT = "高度"
COL_WIDTH = "寬度"
COL_WIDTH_FACTOR = "寬度係數"
COL_TEXT_CENTER_X = "文字中心 X"
COL_TEXT_CENTER_Y = "文字中心 Y"

# === DXF 實體類型名稱 (作為 COL_NAME 欄位的值) ===
ETYPE_LINE = "線"
ETYPE_POLYLINE = "聚合線"
ETYPE_CIRCLE = "圓"
ETYPE_ELLIPSE = "橢圓"
ETYPE_ARC = "弧"
ETYPE_TEXT = "文字"
ETYPE_MTEXT = "多行文字"
ETYPE_HATCH = "填充線"

# === Graph 邊屬性鍵名 ===
EDGE_LENGTH = "length"
EDGE_LINE_ID = "line_id"
EDGE_BLOCK_ID = "block_id"
EDGE_TYPE = "type"

# === 顯示 / 匯出用欄位名 ===
DISP_BLOCK_ID = "Block ID"
DISP_LINE_ID = "Line ID"
DISP_START_X = "Start X"
DISP_START_Y = "Start Y"
DISP_END_X = "End X"
DISP_END_Y = "End Y"
DISP_LENGTH = "Length (mm)"
DISP_COMPONENT = "Component"
DISP_QTY = "QTY"

# === 座標精度 ===
COORD_PRECISION = 4


def round_coord(x: float) -> float:
    """統一座標四捨五入精度。"""
    return round(float(x), COORD_PRECISION)


def make_node(x, y) -> tuple:
    """建立統一精度的 Graph 節點 tuple。"""
    return (round_coord(x), round_coord(y))
