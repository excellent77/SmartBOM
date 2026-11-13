"""單線圖資料夾的批次 BOM 分析服務。

本模組只負責找出合法輸入、執行應用服務與彙整結果；單張圖面的
載入、分析與報表輸出仍由 BomApplicationService 負責。
"""

from collections.abc import Callable
from pathlib import Path
import re

import pandas as pd

from .config import ADJACENCY_OUTPUT_SUFFIX, BOM_OUTPUT_SUFFIX, DETAILS_OUTPUT_SUFFIX
from .service import BomApplicationService


SUPPORTED_DRAWING_EXTENSIONS = {".dxf", ".dwg", ".csv"}
GENERATED_REPORT_SUFFIXES = (
    BOM_OUTPUT_SUFFIX,
    DETAILS_OUTPUT_SUFFIX,
    ADJACENCY_OUTPUT_SUFFIX,
)


def _natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """以人類預期的數字順序排列圖號，例如 2 會在 10 之前。"""
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_single_line_drawings(directory: str | Path) -> list[Path]:
    """找出資料夾內的單線圖輸入檔，排除程式產生的報表。"""
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"找不到資料夾：{folder}")

    drawings = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_DRAWING_EXTENSIONS:
            continue
        if "單線圖" not in path.stem:
            continue
        if any(path.name.casefold().endswith(suffix.casefold()) for suffix in GENERATED_REPORT_SUFFIXES):
            continue
        drawings.append(path)
    return sorted(drawings, key=_natural_sort_key)


def run_directory(
    directory: str | Path = "./data/data_new",
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """批次產生每張單線圖的 BOM、群組明細與鄰接矩陣。

    單一檔案失敗不會中止整批作業，回傳表格會保留失敗類型與訊息。
    """
    drawings = discover_single_line_drawings(directory)
    if not drawings:
        raise FileNotFoundError(f"資料夾內沒有可處理的單線圖：{Path(directory)}")

    service = BomApplicationService.create_default()
    records: list[dict[str, object]] = []
    total = len(drawings)
    for index, source in enumerate(drawings, 1):
        if progress:
            progress(f"[{index}/{total}] 處理 {source.name}")
        try:
            result = service.run(source, write_reports=True)
            output_paths = (
                source.with_name(f"{source.stem}{BOM_OUTPUT_SUFFIX}"),
                source.with_name(f"{source.stem}{DETAILS_OUTPUT_SUFFIX}"),
                source.with_name(f"{source.stem}{ADJACENCY_OUTPUT_SUFFIX}"),
            )
            records.append({
                "輸入檔案": source.name, "狀態": "成功",
                "BOM筆數": len(result.bom), "群組數": len(result.details),
                "BOM檔案": str(output_paths[0]),
                "群組明細檔案": str(output_paths[1]),
                "鄰接矩陣檔案": str(output_paths[2]), "錯誤": "",
            })
        except Exception as exc:  # 批次處理必須繼續下一張圖。
            records.append({
                "輸入檔案": source.name, "狀態": "失敗",
                "BOM筆數": 0, "群組數": 0,
                "BOM檔案": "", "群組明細檔案": "",
                "鄰接矩陣檔案": "", "錯誤": f"{type(exc).__name__}: {exc}",
            })
            if progress:
                progress(f"    失敗：{type(exc).__name__}: {exc}")
    return pd.DataFrame(records)
