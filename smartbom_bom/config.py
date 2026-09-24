"""集中管理 BOM 分析參數與元件顏色映射。

新增設定時只需擴充 :class:\`BomSettings\`，避免演算法模組散落魔術數字。
"""

from dataclasses import dataclass, field

# ===== 可調整參數 =====
COMPONENT_TOLERANCE = 0.5
CONNECTION_TOLERANCE = 0.5
DUPLICATE_TOLERANCE = 0.01
SEGMENT_INTERSECTION_EPSILON = 1e-9
CIRCLE_SAMPLE_STEPS = 72
ARC_SAMPLE_STEPS = 36
MIN_ARC_SAMPLE_STEPS = 8
COMPONENT_SIZE_SCORE_WEIGHT = 100.0
TUBE_COLOR = "顏色_30"
BOM_OUTPUT_SUFFIX = "_bom.csv"
DETAILS_OUTPUT_SUFFIX = "_details.csv"
ADJACENCY_OUTPUT_SUFFIX = "_adjacency.csv"


# 依 GAS零件圖塊對照表_v5 更新；此映射只屬於本程式。
COLOR_COMPONENT_MAP = {
    "顏色_181": {"name": "Ball-valve"},
    "顏色_3": {"name": "Dia-valve"},
    "顏色_24": {"name": "Bellow valve"},
    "顏色_212": {"name": "C.V"},
    "顏色_140": {"name": "3P Regulator"},
    "顏色_5": {"name": "Reducer"},
    "顏色_16": {"name": "ELBOW"},
    "顏色_1": {"name": "Nut(F)"},
    "顏色_8": {"name": "NUT(M)"},
    "顏色_142": {"name": "S-Gland"},
    "顏色_11": {"name": "L-Gland"},
    "顏色_165": {"name": "Gasket"},
    "顏色_171": {"name": "Union R.Tee"},
    "顏色_241": {"name": "Union Tee"},
    "顏色_67": {"name": "Union"},
    "顏色_211": {"name": "Reducer-Union"},
    "顏色_4": {"name": "軟管"},
    "顏色_37": {"name": "Cap"},
    "顏色_6": {"name": "Regulator(SWG)"},
    "顏色_122": {"name": "Regulator(FVCR)"},
    "顏色_33": {"name": "Tee"},
    "顏色_104": {"name": "R.Tee"},
    "顏色_153": {"name": "VCR Tee"},
    "顏色_225": {"name": "VCR R.Tee"},
    "顏色_40": {"name": "SWG Gauge(背接式)"},
    "顏色_192": {"name": "VCR Gauge(背接式)"},
    "顏色_252": {"name": "IGNORE"},  # 表上註記「不須列入計算的材料」，應在後製時直接排除，不當成 BOM 品項
    "顏色_7": {"name": "IGNORE"},  # 表上註記「不須列入計算的材料」，應在後製時直接排除，不當成 BOM 品項
}

TEXT_TYPES = {"文字", "多行文字", "TEXT", "MTEXT"}
GEOMETRY_TYPES = {"線", "聚合線", "圓", "弧", "橢圓", "填充線", "LINE", "CIRCLE", "ARC", "ELLIPSE"}

@dataclass(frozen=True, slots=True)
class BomSettings:
    """一次分析所需的不可變設定，可在測試或不同專案中注入。"""

    component_tolerance: float = COMPONENT_TOLERANCE
    connection_tolerance: float = CONNECTION_TOLERANCE
    duplicate_tolerance: float = DUPLICATE_TOLERANCE
    component_size_score_weight: float = COMPONENT_SIZE_SCORE_WEIGHT
    tube_color: str = TUBE_COLOR
    color_component_map: dict[str, dict[str, str]] = field(
        default_factory=lambda: {key: value.copy() for key, value in COLOR_COMPONENT_MAP.items()}
    )

    def validate(self) -> None:
        """拒絕會讓幾何判定失真的負容許值。"""
        if min(self.component_tolerance, self.connection_tolerance, self.duplicate_tolerance) < 0:
            raise ValueError("容許距離不可小於 0")
