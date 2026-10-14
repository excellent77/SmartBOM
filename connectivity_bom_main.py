"""SmartBOM 人工覆核式 Gradio 操作介面。

UI 只負責上傳、呈現及收集修改；連通塊分析與修正规則由
ConnectivityReviewService 負責，BOM 核心仍由 BomApplicationService 提供。
"""

from pathlib import Path
from typing import Any

import pandas as pd

from smartbom_bom import BomApplicationService, ConnectivityReviewService, ReviewSession
from smartbom_bom.config import ADJACENCY_OUTPUT_SUFFIX, BOM_OUTPUT_SUFFIX, DETAILS_OUTPUT_SUFFIX
from smartbom_bom.visualization import build_debug_figure, figure_to_html


SUPPORTED_EXTENSIONS = {".dxf", ".dwg", ".csv"}
REVIEW_COLUMNS = [
    "優先順序", "連通塊", "風險標記", "尺寸", "規格", "元件摘要",
]
EMPTY_BOM = pd.DataFrame(columns=["品名", "尺寸", "材質", "數量"])
EMPTY_REVIEW = pd.DataFrame(columns=REVIEW_COLUMNS)
EMPTY_DETAILS = pd.DataFrame()
EMPTY_ADJACENCY = pd.DataFrame()
EMPTY_FIGURE = ""


def _output_paths(source: Path) -> list[str]:
    """回傳人工確認後的三份報表路徑。"""
    return [
        str(source.with_name(f"{source.stem}{BOM_OUTPUT_SUFFIX}")),
        str(source.with_name(f"{source.stem}{DETAILS_OUTPUT_SUFFIX}")),
        str(source.with_name(f"{source.stem}{ADJACENCY_OUTPUT_SUFFIX}")),
    ]


def _as_frame(value: object) -> pd.DataFrame:
    """統一不同 Gradio 版本可能回傳的 DataFrame、list 或 dict。"""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame(value)
    if value is None:
        return EMPTY_REVIEW.copy()
    return pd.DataFrame(value, columns=REVIEW_COLUMNS)


def analyze_uploaded_file(file_path: str | Path | None):
    """自動分析上傳檔案，但先不輸出最終 BOM，等待人工覆核。"""
    empty = (
        EMPTY_FIGURE, EMPTY_REVIEW, EMPTY_DETAILS,
        EMPTY_ADJACENCY, EMPTY_BOM, [], None,
    )
    if not file_path:
        return ("請先上傳 DXF、DWG 或向量 CSV。", *empty)

    source = Path(str(file_path))
    if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        return (
            f"❌ 不支援的檔案格式：{source.suffix or '無副檔名'}；請使用 {supported}。",
            *empty,
        )

    try:
        service = BomApplicationService.create_default()
        vectors = service.loader.load(source)
        result = service.analyze(vectors)
        review_service = ConnectivityReviewService(service.report_builder)
        session = review_service.create_session(source, vectors, result)
        review_frame = review_service.to_frame(session)
        figure = figure_to_html(
            build_debug_figure(vectors, result.details, result.bom, session.segments)
        )
        adjacency = result.adjacency.reset_index()
        status = (
            f"✅ **自動分析完成：{source.name}**  \n"
            f"- 找到 {len(session.segments)} 個 Reducer 分段連通塊  \n"
            "- 請依圖面覆核下方「尺寸」與「規格」欄，完成後按 **確認修改並產生 BOM**。"
        )
        return (
            status, figure, review_frame,
            result.details, adjacency, EMPTY_BOM, [], session,
        )
    except Exception as exc:
        return (f"❌ **分析失敗**：{type(exc).__name__}: {exc}", *empty)


def confirm_review(edited_review: object, session: ReviewSession | None):
    """套用人工覆核欄位、重建 Reducer 尺寸及最終 BOM。"""
    if session is None:
        return (
            "❌ 尚未載入可覆核的圖面。",
            EMPTY_FIGURE, EMPTY_REVIEW,
            EMPTY_DETAILS, EMPTY_BOM, [], None,
        )
    try:
        review_service = ConnectivityReviewService()
        result = review_service.apply(session, _as_frame(edited_review))
        BomApplicationService.create_default().writer.write(session.source, result)
        figure = figure_to_html(
            build_debug_figure(session.vectors, result.details, result.bom, session.segments)
        )
        removed_unknown = sum(
            str(group.name).strip().startswith("顏色_") for group in result.groups
        )
        status = (
            "✅ **人工覆核已套用，最終 BOM 已產生。**  \n"
            f"- BOM：{len(result.bom)} 筆  \n"
            f"- 已排除顏色名稱元件組：{removed_unknown} 組  \n"
            "- Gauge、Gauge 相連 Gasket、Xx1/4\" 特殊元件維持固定尺寸規則；"
            "Reducer 尺寸已依修正後相鄰連通塊重新整理。"
        )
        return (
            status, figure, review_service.to_frame(session), result.details, result.bom,
            _output_paths(session.source), session,
        )
    except Exception as exc:
        return (
            f"❌ **套用修改失敗**：{type(exc).__name__}: {exc}",
            EMPTY_FIGURE, _as_frame(edited_review),
            session.result.details, EMPTY_BOM, [], session,
        )


def run_file(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保留無 UI 的自動分析入口；不包含人工覆核。"""
    result = BomApplicationService.create_default().run(file_path, write_reports=True)
    return result.bom, result.details, result.adjacency


def create_interface() -> Any:
    """建立以人工覆核為主流程的 Gradio Blocks。"""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("尚未安裝 Gradio，請先安裝 requirements.txt 內的相依套件。") from exc

    css = """
    #review-table textarea {font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    #confirm-review {min-height:48px;font-weight:700}
    """

    with gr.Blocks(title="SmartBOM 人工覆核與 BOM 產生", css=css) as demo:
        review_state = gr.State(value=None)
        gr.Markdown(
            "# SmartBOM 人工覆核與 BOM 產生\n"
            "上傳圖檔後先檢查 Reducer 分段連通塊；系統不會在人工確認前將自動結果視為最終 BOM。"
        )

        with gr.Row():
            upload = gr.File(
                label="上傳 DXF、DWG 或向量 CSV",
                file_types=[".dxf", ".dwg", ".csv"],
                type="filepath",
            )
            status = gr.Markdown("請上傳檔案開始分析。")

        with gr.Tabs():
            with gr.Tab("① 互動檢視與連通塊覆核"):
                gr.Markdown(
                    "原始圖元線色維持 DXF 顏色；半透明區塊依 Reducer 分段著色。"
                    "滑鼠停留時可查看元件完整資訊與所屬連通塊。"
                )
                debug_plot = gr.HTML(value=EMPTY_FIGURE)
                gr.Markdown(
                    "### 可修改連通塊\n"
                    "表格已依差異數由高至低排序；◆ 小型連通塊代表元件數較少。"
                    "**請只修改「尺寸」與「規格」兩欄**，其餘欄位用於識別與稽核。"
                )
                review_table = gr.Dataframe(
                    headers=REVIEW_COLUMNS, value=EMPTY_REVIEW,
                    interactive=True, wrap=True, elem_id="review-table",
                )
                confirm_button = gr.Button(
                    "確認修改並產生 BOM", variant="primary", elem_id="confirm-review"
                )
                gr.Markdown("### 確認後產生的最終 BOM")
                bom_table = gr.Dataframe(
                    headers=["品名", "尺寸", "材質", "數量"],
                    value=EMPTY_BOM, interactive=False, wrap=True,
                )
                downloads = gr.File(
                    label="下載最終 BOM、明細與鄰接矩陣",
                    file_count="multiple", interactive=False,
                )

            with gr.Tab("② 圖元組明細"):
                details_table = gr.Dataframe(value=EMPTY_DETAILS, interactive=False, wrap=True)
            with gr.Tab("③ 鄰接矩陣"):
                adjacency_table = gr.Dataframe(value=EMPTY_ADJACENCY, interactive=False, wrap=False)

        clear_button = gr.Button("清除目前圖面")
        upload_outputs = [
            status, debug_plot, review_table, details_table,
            adjacency_table, bom_table, downloads, review_state,
        ]
        upload.upload(
            fn=analyze_uploaded_file, inputs=upload, outputs=upload_outputs,
            show_progress="full",
        )
        confirm_button.click(
            fn=confirm_review, inputs=[review_table, review_state],
            outputs=[
                status, debug_plot, review_table,
                details_table, bom_table, downloads, review_state,
            ],
            show_progress="full",
        )
        clear_button.click(
            fn=lambda: (
                "請上傳檔案開始分析。", EMPTY_FIGURE,
                EMPTY_REVIEW, EMPTY_DETAILS, EMPTY_ADJACENCY,
                EMPTY_BOM, [], None,
            ),
            outputs=upload_outputs, queue=False,
        )

    return demo


def main() -> None:
    """啟動 Gradio 網頁介面。"""
    create_interface().queue().launch()


if __name__ == "__main__":
    main()
