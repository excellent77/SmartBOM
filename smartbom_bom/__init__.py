"""SmartBOM 管網 BOM 分析套件的公開 API。"""

from .config import BomSettings
from .geometry import row_to_geometry
from .grouping import build_adjacency_matrix, build_entity_groups
from .io import DefaultVectorDataLoader, CsvReportWriter
from .service import BomAnalysisResult, BomApplicationService, build_network_bom

__all__ = [
    "BomAnalysisResult", "BomApplicationService", "BomSettings", "CsvReportWriter",
    "DefaultVectorDataLoader", "build_adjacency_matrix", "build_entity_groups",
    "build_network_bom", "row_to_geometry",
]
