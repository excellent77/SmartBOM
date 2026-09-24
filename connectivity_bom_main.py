"""SmartBOM 管網 BOM 的 Gradio 使用者介面。

UI 僅負責接收檔案、呈現結果與提供下載；所有工程分析仍由
smartbom_bom.BomApplicationService 完成，避免介面與核心演算法耦合。
"""

from pathlib import Path
from typing import Any

import pandas as pd

from smartbom_bom import BomApplicationService
from smartbom_bom.config import (
    ADJACENCY_OUTPUT_SUFFIX,
    BOM_OUTPUT_SUFFIX,
    DETAILS_OUTPUT_SUFFIX,
)


SUPPORTED_EXTENSIONS = {".dxf", ".dwg", ".csv"}
EMPTY_BOM = pd.DataFrame(columns=["品名", "尺寸", "材質", "數量"])
EMPTY_DETAILS = pd.DataFrame()
EMPTY_ADJACENCY = pd.DataFrame()


def _output_paths(source: Path) -> list[str]:
    """回傳應用服務寫出的三份報表路徑，供 Gradio 下載元件使用。"""
    return [
        str(source.with_name(f"{source.stem}{BOM_OUTPUT_SUFFIX}")),
        str(source.with_name(f"{source.stem}{DETAILS_OUTPUT_SUFFIX}")),
        str(source.with_name(f"{source.stem}{ADJACENCY_OUTPUT_SUFFIX}")),
    ]


def analyze_uploaded_file(
    file_path: str | Path | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """分析 Gradio 上傳檔案並回傳畫面資料及下載路徑。

    此函式不依賴 Gradio 型別，因此可以獨立做單元測試，也能被其他 UI 重用。
    """
    if not file_path:
        return "請先上傳 DXF、DWG 或向量 CSV。", EMPTY_BOM, EMPTY_DETAILS, EMPTY_ADJACENCY, []

    source = Path(str(file_path))
    if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        return (
            f"不支援的檔案格式：{source.suffix or '無副檔名'}；請使用 {supported}。",
            EMPTY_BOM,
            EMPTY_DETAILS,
            EMPTY_ADJACENCY,
            [],
        )

    try:
        # 上傳檔位於 Gradio 暫存目錄；報表寫在同一目錄，不會覆寫專案原始資料。
        result = BomApplicationService.create_default().run(source, write_reports=True)
        removed_rows = result.details.attrs.get("removed_duplicate_rows", [])
        adjacency_for_display = result.adjacency.reset_index()
        status = (
            f"✅ **分析完成：{source.name}**  \n"
            f"- BOM：{len(result.bom)} 筆  \n"
            f"- 圖元組明細：{len(result.details)} 組  \n"
            f"- 鄰接矩陣：{result.adjacency.shape[0]} × {result.adjacency.shape[1]}  \n"
            f"- 去除同色重複線段：{len(removed_rows)} 條"
        )
        return (
            status,
            result.bom,
            result.details,
            adjacency_for_display,
            _output_paths(source),
        )
    except Exception as exc:
        return (
            f"❌ **分析失敗**：{type(exc).__name__}: {exc}",
            EMPTY_BOM,
            EMPTY_DETAILS,
            EMPTY_ADJACENCY,
            [],
        )


def run_file(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保留無 UI 的程式化入口，回傳 BOM、明細與鄰接矩陣。"""
    result = BomApplicationService.create_default().run(file_path, write_reports=True)
    return result.bom, result.details, result.adjacency


def create_interface() -> Any:
    """建立 Gradio Blocks；延遲 import 讓核心模組不必依賴 Gradio 才能載入。"""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "尚未安裝 Gradio，請先安裝 requirements.txt 內的相依套件。"
        ) from exc

    with gr.Blocks(title="SmartBOM 管網 BOM 分析") as demo:
        gr.Markdown(
            "# SmartBOM 管網 BOM 分析\n"
            "上傳 DXF、DWG 或 dxf_reader 產生的向量 CSV，系統會自動分析並產生 BOM。"
        )

        with gr.Row():
            upload = gr.File(
                label="上傳工程圖檔",
                file_types=[".dxf", ".dwg", ".csv"],
                type="filepath",
            )
            status = gr.Markdown("請上傳檔案開始分析。")

        with gr.Tabs():
            with gr.Tab("BOM"):
                gr.Markdown("### 最終 BOM")
                bom_table = gr.Dataframe(
                    headers=["品名", "尺寸", "材質", "數量"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("圖元組明細"):
                details_table = gr.Dataframe(interactive=False, wrap=True)
            with gr.Tab("鄰接矩陣"):
                adjacency_table = gr.Dataframe(interactive=False, wrap=False)

        downloads = gr.File(
            label="下載分析結果（BOM、明細、鄰接矩陣）",
            file_count="multiple",
            interactive=False,
        )
        clear_button = gr.Button("清除結果")

        outputs = [status, bom_table, details_table, adjacency_table, downloads]
        upload.upload(
            fn=analyze_uploaded_file,
            inputs=upload,
            outputs=outputs,
            show_progress="full",
        )
        clear_button.click(
            fn=lambda: (
                "請上傳檔案開始分析。",
                EMPTY_BOM,
                EMPTY_DETAILS,
                EMPTY_ADJACENCY,
                [],
            ),
            outputs=outputs,
            queue=False,
        )

    return demo


def main() -> None:
    """啟動 Gradio 網頁介面。"""
    create_interface().queue().launch()


if __name__ == "__main__":
    main()
