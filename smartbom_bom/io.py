"""輸入與輸出邊界；核心演算法不直接接觸檔案系統。"""

from pathlib import Path
from typing import Protocol
import pandas as pd
from dxf_reader import read_dwg
from .config import ADJACENCY_OUTPUT_SUFFIX, BOM_OUTPUT_SUFFIX, DETAILS_OUTPUT_SUFFIX
from .models import BomAnalysisResult

class VectorDataLoader(Protocol):
    """向量資料來源介面，便於替換 DXF reader 或測試替身。"""
    def load(self, path: str | Path) -> pd.DataFrame: ...

class ReportWriter(Protocol):
    """分析結果輸出介面。"""
    def write(self, input_path: str | Path, result: BomAnalysisResult) -> tuple[Path, Path, Path]: ...

class DefaultVectorDataLoader:
    """支援 DXF、DWG 與既有向量 CSV 的預設載入器。"""
    CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp950")

    def load(self, path: str | Path) -> pd.DataFrame:
        input_path = Path(path)
        extension = input_path.suffix.casefold()
        if extension in {".dxf", ".dwg"}:
            return read_dwg(str(input_path))
        if extension == ".csv":
            errors = []
            for encoding in self.CSV_ENCODINGS:
                try:
                    return pd.read_csv(input_path, encoding=encoding, low_memory=False)
                except UnicodeDecodeError as exc:
                    errors.append(f"{encoding}: {exc}")
            raise UnicodeError("無法判斷 CSV 編碼：" + "; ".join(errors))
        raise ValueError(f"不支援的輸入格式：{extension}；請使用 .dxf、.dwg 或 .csv")

class CsvReportWriter:
    """將 BOM、群組明細與鄰接矩陣寫為 UTF-8-SIG CSV。"""
    def write(self, input_path: str | Path, result: BomAnalysisResult) -> tuple[Path, Path, Path]:
        source = Path(input_path)
        paths = (
            source.with_name(f"{source.stem}{BOM_OUTPUT_SUFFIX}"),
            source.with_name(f"{source.stem}{DETAILS_OUTPUT_SUFFIX}"),
            source.with_name(f"{source.stem}{ADJACENCY_OUTPUT_SUFFIX}"),
        )
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        result.bom.to_csv(paths[0], index=False, encoding="utf-8-sig")
        result.details.to_csv(paths[1], index=False, encoding="utf-8-sig")
        result.adjacency.to_csv(paths[2], encoding="utf-8-sig")
        return paths

def read_vector_csv(path: str | Path) -> pd.DataFrame:
    return DefaultVectorDataLoader().load(path)

def load_vector_data(path: str | Path) -> pd.DataFrame:
    return DefaultVectorDataLoader().load(path)
