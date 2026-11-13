"""SmartBOM 人工覆核式 Gradio 操作介面。

UI 只負責上傳、呈現及收集修改；連通塊分析與修正规則由
ConnectivityReviewService 負責，BOM 核心仍由 BomApplicationService 提供。
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from smartbom_bom import BomApplicationService, ConnectivityReviewService, ReviewSession
from smartbom_bom.config import (
    ADJACENCY_OUTPUT_SUFFIX, BOM_OUTPUT_SUFFIX, COLOR_COMPONENT_MAP,
    COMPONENT_TYPES, COMPONENT_TYPE_GENERAL, COMPONENT_TYPE_REDUCER,
    COMPONENT_TYPE_TUBE, DETAILS_OUTPUT_SUFFIX, BomSettings,
)
from smartbom_bom.geometry import row_to_geometry
from smartbom_bom.predicates import infer_component_type
from smartbom_bom.visualization import build_debug_figure, figure_to_html


SUPPORTED_EXTENSIONS = {".dxf", ".dwg", ".csv"}
REVIEW_COLUMNS = [
    "優先順序", "連通塊", "風險標記", "尺寸", "規格", "元件摘要",
]
COMPONENT_MAPPING_COLUMNS = ["元件顏色", "元件名稱", "元件類型"]
COMPONENT_TYPE_OPTIONS = [
    COMPONENT_TYPE_TUBE, COMPONENT_TYPE_REDUCER, COMPONENT_TYPE_GENERAL,
]
EMPTY_BOM = pd.DataFrame(columns=["品名", "尺寸", "材質", "數量"])
EMPTY_REVIEW = pd.DataFrame(columns=REVIEW_COLUMNS)
EMPTY_DETAILS = pd.DataFrame()
EMPTY_ADJACENCY = pd.DataFrame()
EMPTY_FIGURE = ""
EMPTY_COMPONENT_MAPPING = pd.DataFrame(columns=COMPONENT_MAPPING_COLUMNS)
UI_CSS = """
#review-table textarea {font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
#confirm-review {min-height:48px;font-weight:700}
"""


@dataclass(slots=True)
class PreparedUpload:
    """已載入但尚未進行 BOM 分析的圖面。"""

    source: Path
    vectors: pd.DataFrame
    colors: tuple[str, ...]


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


def _color_sort_key(color: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"顏色_(\d+)", str(color).strip())
    return (0, int(match.group(1))) if match else (1, str(color))


def _component_mapping_for_colors(colors: tuple[str, ...]) -> pd.DataFrame:
    """已知顏色預填現行映射；未知顏色名稱留空等待使用者。"""
    rows = []
    for color in colors:
        configured = COLOR_COMPONENT_MAP.get(color, {})
        name = str(configured.get("name", "")).strip()
        component_type = configured.get("type") or infer_component_type(name, color)
        rows.append({
            "元件顏色": color,
            "元件名稱": name,
            "元件類型": component_type,
        })
    return pd.DataFrame(rows, columns=COMPONENT_MAPPING_COLUMNS)


def prepare_uploaded_file(file_path: str | Path | None):
    """只載入圖面並建立元件顏色對照表，尚不執行後續分析。"""
    if not file_path:
        return "請先上傳 DXF、DWG 或向量 CSV。", EMPTY_COMPONENT_MAPPING, None

    source = Path(str(file_path))
    if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        return f"❌ 不支援的檔案格式：{source.suffix or '無副檔名'}；請使用 {supported}。", EMPTY_COMPONENT_MAPPING, None

    try:
        service = BomApplicationService.create_default()
        vectors = service.loader.load(source)
        colors = tuple(sorted({
            geometry.color
            for index, row in vectors.iterrows()
            if (geometry := row_to_geometry(int(index), row)) is not None
        }, key=_color_sort_key))
        if not colors:
            raise ValueError("圖面中找不到可用幾何圖元")
        mapping = _component_mapping_for_colors(colors)
        unknown = int((mapping["元件名稱"].str.strip() == "").sum())
        status = (
            f"✅ **圖面已載入：{source.name}**  \n"
            f"- 找到 {len(colors)} 種圖元顏色  \n"
            f"- 其中 {unknown} 種尚無元件名稱  \n"
            "- 請完成第一頁的每一格，再按 **確認元件對照並開始分析**。"
        )
        return status, mapping, PreparedUpload(source, vectors, colors)
    except Exception as exc:
        return f"❌ **載入失敗**：{type(exc).__name__}: {exc}", EMPTY_COMPONENT_MAPPING, None


def _validated_component_map(value: object, expected_colors: tuple[str, ...]) -> dict[str, dict[str, str]]:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("元件對照資料必須是 DataFrame")
    frame = value.copy().fillna("")
    missing_columns = [column for column in COMPONENT_MAPPING_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("元件對照表缺少欄位：" + "、".join(missing_columns))
    frame = frame[COMPONENT_MAPPING_COLUMNS].astype(str)
    for column in COMPONENT_MAPPING_COLUMNS:
        frame[column] = frame[column].str.strip()
    if frame.empty:
        raise ValueError("元件對照表不可為空")
    empty_cells = [
        f"第{index + 1}列的{column}"
        for index, row in frame.iterrows()
        for column in COMPONENT_MAPPING_COLUMNS if not row[column]
    ]
    if empty_cells:
        raise ValueError("以下欄位尚未填寫：" + "、".join(empty_cells[:20]))
    duplicates = frame[frame["元件顏色"].duplicated()]["元件顏色"].tolist()
    if duplicates:
        raise ValueError("元件顏色重複：" + "、".join(duplicates))
    supplied = set(frame["元件顏色"])
    expected = set(expected_colors)
    if supplied != expected:
        raise ValueError(
            f"元件顏色與圖面不一致；缺少：{sorted(expected - supplied)}；"
            f"多出：{sorted(supplied - expected)}"
        )
    invalid_types = sorted(set(frame["元件類型"]) - COMPONENT_TYPES)
    if invalid_types:
        raise ValueError(
            "元件類型只能是 Tube類、Reducer類或一般元件；無效值："
            + "、".join(invalid_types)
        )
    return {
        row["元件顏色"]: {
            "name": row["元件名稱"], "type": row["元件類型"],
        }
        for _, row in frame.iterrows()
    }


def analyze_component_mapping(component_mapping: object, prepared: PreparedUpload | None):
    """驗證使用者完成的對照表，通過後才執行 BOM 分析。"""
    empty = (EMPTY_FIGURE, EMPTY_REVIEW, EMPTY_DETAILS, EMPTY_ADJACENCY, EMPTY_BOM, [], None)
    if prepared is None:
        return ("❌ 尚未載入可分析的圖面。", *empty)
    try:
        component_map = _validated_component_map(component_mapping, prepared.colors)
        settings = BomSettings(color_component_map=component_map)
        service = BomApplicationService.create_default(settings)
        result = service.analyze(prepared.vectors)
        review_service = ConnectivityReviewService(service.report_builder)
        session = review_service.create_session(prepared.source, prepared.vectors, result)
        review_frame = review_service.to_frame(session)
        figure = figure_to_html(
            build_debug_figure(prepared.vectors, result.details, result.bom, session.segments)
        )
        adjacency = result.adjacency.reset_index()
        status = (
            f"✅ **元件對照已確認，自動分析完成：{prepared.source.name}**  \n"
            f"- 找到 {len(session.segments)} 個 Reducer 分段連通塊  \n"
            "- 請依圖面覆核下方「尺寸」與「規格」欄，完成後按 **確認修改並產生 BOM**。"
        )
        return (
            status, figure, review_frame,
            result.details, adjacency, EMPTY_BOM, [], session,
        )
    except Exception as exc:
        return (f"❌ **分析失敗**：{type(exc).__name__}: {exc}", *empty)


def prepare_interface_file(file_path: str | Path | None):
    """上傳新圖面時同時清空上一次分析結果。"""
    status, _mapping, prepared = prepare_uploaded_file(file_path)
    return (
        status, prepared, EMPTY_FIGURE, EMPTY_REVIEW,
        EMPTY_DETAILS, EMPTY_ADJACENCY, EMPTY_BOM, [], None,
    )


def analyze_component_inputs(prepared: PreparedUpload | None, *values: object):
    """將動態產生的名稱輸入與類型選單組回對照表。"""
    if prepared is None:
        return analyze_component_mapping(EMPTY_COMPONENT_MAPPING, None)
    expected_values = len(prepared.colors) * 2
    if len(values) != expected_values:
        empty = (EMPTY_FIGURE, EMPTY_REVIEW, EMPTY_DETAILS, EMPTY_ADJACENCY, EMPTY_BOM, [], None)
        return (
            f"❌ **分析失敗**：元件對照輸入數量異常，"
            f"預期 {expected_values} 格，實際 {len(values)} 格。",
            *empty,
        )
    rows = []
    for index, color in enumerate(prepared.colors):
        rows.append({
            "元件顏色": color,
            "元件名稱": values[index * 2],
            "元件類型": values[index * 2 + 1],
        })
    return analyze_component_mapping(pd.DataFrame(rows), prepared)


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
            "- Gauge 與 Gauge 相連 Gasket 維持 1/4\" 固定尺寸；"
            "Reducer類尺寸已依所有方向的兩種連通塊尺寸重建。"
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


def create_interface() -> Any:
    """建立以人工覆核為主流程的 Gradio Blocks。"""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("尚未安裝 Gradio，請先安裝 requirements.txt 內的相依套件。") from exc

    with gr.Blocks(title="SmartBOM 人工覆核與 BOM 產生") as demo:
        prepared_state = gr.State(value=None)
        review_state = gr.State(value=None)
        gr.Markdown(
            "# SmartBOM 人工覆核與 BOM 產生\n"
            "上傳圖檔後先確認元件顏色對照，再進行連通塊分析與人工覆核。"
        )

        with gr.Row():
            upload = gr.File(
                label="上傳 DXF、DWG 或向量 CSV",
                file_types=[".dxf", ".dwg", ".csv"],
                type="filepath",
            )
            status = gr.Markdown("請上傳檔案開始分析。")

        with gr.Tabs():
            with gr.Tab("① 元件顏色與類型"):
                gr.Markdown(
                    "請確認這次圖面中每種顏色的元件名稱與類型。"
                    "已存在於 `COLOR_COMPONENT_MAP` 的顏色會自動預填；新顏色的名稱留空。\n\n"
                    "「元件類型」請由下拉選單選擇 **Tube類**、**Reducer類** 或 **一般元件**。"
                    "Tube類會使用管線文字配對；Reducer類會作為連通塊的尺寸邊界。"
                )

                @gr.render(inputs=prepared_state)
                def render_component_mapping(prepared: PreparedUpload | None):
                    if prepared is None:
                        gr.Markdown("上傳圖檔後，這裡會顯示本次圖面的元件對照表。")
                        return
                    defaults = _component_mapping_for_colors(prepared.colors).set_index("元件顏色")
                    editors = []
                    with gr.Row(key="component-mapping-header"):
                        with gr.Column(scale=1):
                            gr.Markdown("**元件顏色**")
                        with gr.Column(scale=2):
                            gr.Markdown("**元件名稱**")
                        with gr.Column(scale=2):
                            gr.Markdown("**元件類型**")
                    for index, color in enumerate(prepared.colors):
                        default = defaults.loc[color]
                        with gr.Row(key=f"component-mapping-{index}"):
                            gr.Textbox(
                                value=color, interactive=False, show_label=False,
                                container=False, scale=1, key=f"component-color-{color}",
                            )
                            name = gr.Textbox(
                                value=default["元件名稱"], show_label=False,
                                container=False, scale=2, key=f"component-name-{color}",
                            )
                            component_type = gr.Radio(
                                choices=COMPONENT_TYPE_OPTIONS,
                                value=default["元件類型"],
                                interactive=True, show_label=False,
                                container=False, scale=2,
                                key=f"component-type-{color}",
                            )
                        editors.extend([name, component_type])
                    analyze_button = gr.Button(
                        "確認元件對照並開始分析",
                        variant="primary", key="analyze-component-mapping",
                    )
                    analyze_button.click(
                        fn=analyze_component_inputs,
                        inputs=[prepared_state, *editors],
                        outputs=[
                            status, debug_plot, review_table, details_table,
                            adjacency_table, bom_table, downloads, review_state,
                        ],
                        show_progress="full", key="analyze-component-mapping-click",
                    )

            with gr.Tab("② 互動檢視與連通塊覆核"):
                gr.Markdown(
                    "原始圖元線色維持 DXF 顏色，連通塊仍依原配色標示；"
                    "圖上黑色 C01、C02 標籤代表 Reducer 分段連通塊。"
                    "**黃框黃底**代表小型連通塊；尺寸／材質差異不再改變框線或底色。"
                    "紅色「⚠N」標籤仍代表該連通塊差異數，可用圖面左上角「顯示差異」開關切換；"
                    "滑鼠停留時可查看元件完整資訊、所屬連通塊與差異內容。"
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

            with gr.Tab("③ 圖元組明細"):
                details_table = gr.Dataframe(value=EMPTY_DETAILS, interactive=False, wrap=True)
            with gr.Tab("④ 鄰接矩陣"):
                adjacency_table = gr.Dataframe(value=EMPTY_ADJACENCY, interactive=False, wrap=False)

        clear_button = gr.Button("清除目前圖面")
        upload_outputs = [
            status, prepared_state, debug_plot, review_table,
            details_table, adjacency_table, bom_table, downloads, review_state,
        ]
        upload.upload(
            fn=prepare_interface_file, inputs=upload, outputs=upload_outputs,
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
                "請上傳檔案開始分析。", None,
                EMPTY_FIGURE, EMPTY_REVIEW, EMPTY_DETAILS,
                EMPTY_ADJACENCY, EMPTY_BOM, [], None,
            ),
            outputs=upload_outputs, queue=False,
        )

    return demo


def main() -> None:
    """啟動 Gradio 網頁介面。"""
    create_interface().queue().launch(css=UI_CSS)


if __name__ == "__main__":
    main()
