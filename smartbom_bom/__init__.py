"""SmartBOM 管網 BOM 分析套件的公開 API。"""

from .config import BomSettings
from .geometry import row_to_geometry
from .grouping import build_adjacency_matrix, build_entity_groups
from .io import DefaultVectorDataLoader, CsvReportWriter
from .service import BomAnalysisResult, BomApplicationService, build_network_bom
from .review import ConnectivityReviewService, ReviewSession, ReviewSegment
from .batch import discover_single_line_drawings, run_directory

__all__ = [
    "BomAnalysisResult", "BomApplicationService", "BomSettings", "CsvReportWriter",
    "ConnectivityReviewService", "ReviewSession", "ReviewSegment",
    "DefaultVectorDataLoader", "build_adjacency_matrix", "build_entity_groups",
    "build_network_bom", "discover_single_line_drawings", "row_to_geometry",
    "run_directory",
]
