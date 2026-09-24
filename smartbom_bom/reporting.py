"""將領域模型投影為 BOM 與稽核明細。"""

import math
from typing import Sequence
import pandas as pd
from .models import EntityGroup

def _format_number(value: float) -> str:
    return str(int(value)) if math.isclose(value, round(value)) else f"{value:g}"

def make_reports(groups: Sequence[EntityGroup]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for group in groups:
        detail_rows.append({
            "群組編號": group.group_id,
            "品名": group.name,
            "顏色": group.color,
            "尺寸": group.size,
            "材質": group.material,
            "原始數量": group.quantity,
            "配對文字": group.matched_text,
            "中心 X": group.center[0],
            "中心 Y": group.center[1],
            "圖元數": len(group.geometries),
            "來源列號": ",".join(map(str, group.source_rows)),
            "備註": "；".join(group.notes),
        })
    details = pd.DataFrame(detail_rows)

    aggregated: dict[tuple[str, str, str], float] = {}
    for group in groups:
        if group.name == "IGNORE":
            continue
        key = (group.name, group.size, group.material)
        if group.name in {"Tube", "Coil Tube", "Hose", "軟管"}:
            # 沒有合法長度文字的圖元數量為 0，不以幾何長度臆測。
            if group.quantity <= 0:
                continue
            aggregated[key] = aggregated.get(key, 0.0) + group.quantity
        else:
            aggregated[key] = aggregated.get(key, 0.0) + 1.0

    bom_rows = []
    for (name, size, material), quantity in aggregated.items():
        display_quantity = f"{_format_number(quantity)}M" if name in {"Tube", "Coil Tube", "Hose", "軟管"} else int(quantity)
        bom_rows.append({"品名": name, "尺寸": size, "材質": material, "數量": display_quantity})
    return pd.DataFrame(bom_rows, columns=["品名", "尺寸", "材質", "數量"]), details



class BomReportBuilder:
    """建立最終 BOM 與逐群組明細表。"""
    def build(self, groups: Sequence[EntityGroup]):
        return make_reports(groups)
