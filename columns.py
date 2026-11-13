"""集中定義 dxf_reader 產生的向量 DataFrame 欄位名稱。"""

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

# === 座標精度 ===
COORD_PRECISION = 4


def round_coord(x: float) -> float:
    """統一座標四捨五入精度。"""
    return round(float(x), COORD_PRECISION)
