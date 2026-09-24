"""BOM 應用服務：只負責串聯各個單一責任元件。"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .config import BomSettings
from .grouping import EntityGroupingService
from .io import CsvReportWriter, DefaultVectorDataLoader, ReportWriter, VectorDataLoader
from .models import BomAnalysisResult
from .reporting import BomReportBuilder
from .rules import BomRuleEngine
from .text import TextSpecificationService
from .optimizer import TubeAssignmentOptimizer

@dataclass(slots=True)
class BomApplicationService:
    """BOM 分析用例的協調器；所有外部服務都可由建構式注入。"""

    settings: BomSettings
    loader: VectorDataLoader
    writer: ReportWriter
    grouping: EntityGroupingService
    text_service: TextSpecificationService
    rule_engine: BomRuleEngine
    report_builder: BomReportBuilder

    @classmethod
    def create_default(cls, settings: BomSettings | None = None) -> "BomApplicationService":
        resolved = settings or BomSettings()
        return cls(
            settings=resolved, loader=DefaultVectorDataLoader(),
            writer=CsvReportWriter(),
            grouping=EntityGroupingService(
                tube_color=resolved.tube_color,
                component_map=resolved.color_component_map,
            ),
            text_service=TextSpecificationService(),
            rule_engine=BomRuleEngine(TubeAssignmentOptimizer(
                resolved.component_size_score_weight
            )),
            report_builder=BomReportBuilder(),
        )

    def analyze(self, df: pd.DataFrame) -> BomAnalysisResult:
        """分析已載入的向量表，不進行任何檔案寫入。"""
        self.settings.validate()
        groups, removed_rows = self.grouping.build_groups(
            df, self.settings.component_tolerance, self.settings.duplicate_tolerance
        )
        matrix, adjacency = self.grouping.build_adjacency(groups, self.settings.connection_tolerance)
        texts = self.text_service.extract(df)
        self.rule_engine.apply(groups, adjacency, texts)
        bom, details = self.report_builder.build(groups)
        details.attrs["removed_duplicate_rows"] = removed_rows
        labels = [f"G{group.group_id}:{group.name}" for group in groups]
        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        matrix_df.index.name = "群組"
        return BomAnalysisResult(bom=bom, details=details, adjacency=matrix_df)

    def run(self, file_path: str | Path, write_reports: bool = True) -> BomAnalysisResult:
        """載入一個向量檔、執行分析，並依需求輸出 CSV。"""
        source = Path(file_path)
        if not source.is_file():
            raise FileNotFoundError(f"找不到輸入檔案：{source}")
        result = self.analyze(self.loader.load(source))
        if write_reports:
            self.writer.write(source, result)
        return result

def build_network_bom(df: pd.DataFrame, component_tolerance=None, connection_tolerance=None, duplicate_tolerance=None):
    """保留舊 API；新程式建議直接使用 BomApplicationService。"""
    defaults = BomSettings()
    settings = BomSettings(
        component_tolerance=defaults.component_tolerance if component_tolerance is None else component_tolerance,
        connection_tolerance=defaults.connection_tolerance if connection_tolerance is None else connection_tolerance,
        duplicate_tolerance=defaults.duplicate_tolerance if duplicate_tolerance is None else duplicate_tolerance,
    )
    result = BomApplicationService.create_default(settings).analyze(df)
    return result.bom, result.details, result.adjacency
