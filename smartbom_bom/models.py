"""不依賴流程實作的領域資料模型。"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Geometry:
    row_index: int
    color: str
    kind: str
    points: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.points[:, 0].min()), float(self.points[:, 1].min()),
            float(self.points[:, 0].max()), float(self.points[:, 1].max()),
        )


@dataclass
class EntityGroup:
    group_id: int
    name: str
    color: str
    geometries: list[Geometry]
    center: tuple[float, float]
    source_rows: list[int]
    size: str = ""
    material: str = ""
    quantity: float = 1.0
    matched_text: str = ""
    notes: list[str] = field(default_factory=list)

@dataclass(slots=True)
class BomAnalysisResult:
    """完整分析結果，由應用服務交給輸出介面。"""

    bom: object
    details: object
    adjacency: object
