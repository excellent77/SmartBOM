"""舊版相容入口。

核心實作已移至 smartbom_bom 套件。既有程式可繼續從本模組匯入函式；
新程式建議使用 BomApplicationService 或 connectivity_bom_main.main。
"""

from pathlib import Path
from smartbom_bom.config import *
from smartbom_bom.geometry import (
    GeometryService, cluster_geometries, component_center, geometry_distance,
    remove_duplicate_lines, row_to_geometry,
)
from smartbom_bom.grouping import (
    EntityGroupingService, build_adjacency_matrix, build_entity_groups, groups_connected,
)
from smartbom_bom.io import DefaultVectorDataLoader, load_vector_data, read_vector_csv
from smartbom_bom.models import EntityGroup, Geometry
from smartbom_bom.optimizer import TubeAssignmentOptimizer, assign_tube_descriptions
from smartbom_bom.reporting import BomReportBuilder, make_reports
from smartbom_bom.rules import BomRuleEngine, fill_by_integer_programming
from smartbom_bom.service import BomApplicationService, build_network_bom
from smartbom_bom.text import (
    TextSpecificationService, assign_nearest_text_to_groups, clean_dxf_text,
    extract_texts,
)

def main(file_path: str) -> int:
    """分析單一檔案並輸出 BOM、明細與鄰接矩陣 CSV。"""
    result = BomApplicationService.create_default().run(file_path, write_reports=True)
    source = Path(file_path)
    removed = result.details.attrs.get("removed_duplicate_rows", [])
    print(f"BOM：{source.with_name(source.stem + BOM_OUTPUT_SUFFIX)}（{len(result.bom)} 筆）")
    print(f"群組明細：{source.with_name(source.stem + DETAILS_OUTPUT_SUFFIX)}（{len(result.details)} 組）")
    print(f"鄰接矩陣：{source.with_name(source.stem + ADJACENCY_OUTPUT_SUFFIX)}（{result.adjacency.shape[0]} x {result.adjacency.shape[1]}）")
    print(f"去除同色重複線段：{len(removed)} 條（原始列：{removed}）")
    return 0
