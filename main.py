import pandas as pd
import gradio as gr
from units_process import calculate_line, visualize_graph_components, find_corner_points, generate_component_report, export_to_csv
from dxf_reader import read_dwg
from connectivity_OR_bom import build_entity_groups, COMPONENT_TOLERANCE, DUPLICATE_TOLERANCE
from columns import (
    EDGE_BLOCK_ID, EDGE_LINE_ID, EDGE_LENGTH,
    DISP_BLOCK_ID, DISP_LINE_ID, DISP_START_X, DISP_START_Y,
    DISP_END_X, DISP_END_Y, DISP_LENGTH,
    make_node,
)

# 全域變數用於存儲目前處理中的圖形數據
current_data = {
    "graph": None, 
    "df": None, 
    "corner_points": [],
    "component_df": None
}

COMPONENT_LIST = [
    "Ball Valve",
    "Diaphragm Valve",
    "Check Valve (CV)",
    "Regulagtor",
    "3P Regulagtor",
    "Tee",
    "Reducer",
    "ELBOW",
    "NUT(F)",
    "NUT(M)",
    "S Gland",
    "L Gland",
    "GASKET",
    "VCR Tee",
    "Union R.Tee",
    "Union Tee",
    "Gauge",
    "Union",
    "Reducer Union",
    "Hose",
]

def process_file(file):
    """
    處理上傳的 CAD DWG 檔案，啟動管網拓撲分析、元件偵測與轉角點辨識。

    Args:
        file (tempfile._TemporaryFileWrapper): Gradio 傳入的暫存檔案物件。

    Returns:
        tuple: 包含 UI 更新所需的圖表、數值與資料表數據。
    """
    if file is None:
        return None, None, None, None, gr.Dropdown(choices=[], value=[])
    df = read_dwg(file.name)
    current_data["df"] = df
    graph = calculate_line(df)
    
    # block_id 已於 build_graph() 中自動設定
            
    current_data["graph"] = graph
    corners = find_corner_points(graph)
    current_data["corner_points"] = corners
    current_data["component_df"] = generate_component_report(df, COMPONENT_LIST)
    current_data["components_info"] = []
    
    gr.Info("資料讀取與管網分析成功！")
    return update_ui()

def load_components(progress=gr.Progress(track_tqdm=False)):
    """依據 BOM 報表統計數量與位置萃取元件"""
    df = current_data.get("df")
    if df is None:
        gr.Warning("請先上傳檔案")
        return update_ui() if current_data.get("graph") else (None, 0, None, None, gr.Dropdown(choices=[], value=[]))
    
    try:
        from columns import COL_NAME, COL_COUNT
        
        # 取得報表的數量統計
        comp_df = current_data.get("component_df")
        if comp_df is None or comp_df.empty:
            raise ValueError("找不到元件報表資料")
            
        report_counts = dict(zip(comp_df.iloc[:, 0], comp_df.iloc[:, 1]))
        
        # 定義與 generate_component_report 中相同的對應關係
        REPORT_MAPPING = {
            "Ball Valve": ("ball_value", "顏色_181"),
            "Diaphragm Valve": ("dia valve", "顏色_3"),
            "Check Valve (CV)": ("cv", "顏色_212"),
            "Regulagtor": ("regulator", "顏色_6"),
            "3P Regulagtor": ("3p_regulator", "顏色_140"),
            "Tee": ("Tee", "顏色_134"),
            "Reducer": ("reducer", "顏色_5"),
            "ELBOW": ("elbow", "顏色_16"),
            "NUT(F)": ("NUT(F)", "顏色_1"),
            "NUT(M)": ("NUT(M)", "顏色_8"),
            "S Gland": ("S Gland", "顏色_142"),
            "L Gland": ("L Gland", "顏色_11"),
            "GASKET": ("GASKET", "顏色_165"),
            "VCR Tee": ("VCR Tee", "顏色_7"),
            "VCR 正 Tee": ("VCR 正 Tee", "顏色_84"),
            "Union R.Tee": ("Union R.Tee", "顏色_171"),
            "Union Tee": ("Union Tee", "顏色_241"),
            "Gauge": ("Gauge", "顏色_40"),
            "Union": ("Union", "顏色_67"),
            "Reducer Union": ("Reducer Union", "顏色_211"),
            "Hose": ("hose", "顏色_4"),
            "CAP(母塞頭)": ("CAP", "顏色_37")
        }
        
        components_info = []
        df_blocks = df[df[COL_NAME].notna()]
        categories = df_blocks[COL_NAME].unique()
        
        progress(0.3, desc="正在比對報表資料與圖元座標...")
        
        for report_name, (block_search_name, fallback_color) in REPORT_MAPPING.items():
            expected_qty = int(report_counts.get(report_name, 0))
            if expected_qty <= 0:
                continue
                
            pts = []
            # 1. 依照 check_block 邏輯尋找圖塊座標
            found_block = False
            for val in categories:
                if block_search_name.lower() in str(val).lower():
                    subset = df_blocks[df_blocks[COL_NAME] == val]
                    for _, row in subset.iterrows():
                        x, y = row.get("X"), row.get("Y")
                        if pd.notna(x) and pd.notna(y):
                            pts.append((float(x), float(y)))
                    found_block = True
                    break
            
            # 2. 如果報表有數量但沒找到圖塊 (代表報表是使用 fallback 算出來的)，我們使用顏色抓取中心點
            if not found_block and fallback_color:
                # 簡單抓取該顏色的圖元座標，再用 K-Means 聚類成報表指定的數量
                from connectivity_OR_bom import row_to_geometry
                geom_pts = []
                for idx, row in df.iterrows():
                    geom = row_to_geometry(int(idx), row)
                    if geom and geom.color == fallback_color:
                        geom_pts.append(((geom.bbox[0] + geom.bbox[2])/2, (geom.bbox[1] + geom.bbox[3])/2))
                
                if geom_pts:
                    if len(geom_pts) <= expected_qty:
                        pts.extend(geom_pts)
                    else:
                        # 聚類將多個線條中心點還原成 expected_qty 個元件座標
                        import numpy as np
                        from scipy.cluster.vq import kmeans2
                        data = np.array(geom_pts, dtype=float)
                        # minit='points' 避免警告，並從資料點中隨機選取初始中心
                        centroids, _ = kmeans2(data, expected_qty, minit='points')
                        for cx, cy in centroids:
                            pts.append((float(cx), float(cy)))

            # 將找出的點加入總表
            for x, y in pts:
                components_info.append((x, y, report_name))
        
        progress(0.95, desc=f"完成！依照報表對齊 {len(components_info)} 個元件")
        print(f"[INFO] 依照報表篩選出 {len(components_info)} 個元件")
        if components_info:
            all_cx = [p[0] for p in components_info]
            all_cy = [p[1] for p in components_info]
            print(f"[INFO] 元件 X 範圍: {min(all_cx):.1f} ~ {max(all_cx):.1f}")
            print(f"[INFO] 元件 Y 範圍: {min(all_cy):.1f} ~ {max(all_cy):.1f}")
            # 管線座標範圍
            G = current_data.get("graph")
            if G and G.nodes():
                gx = [n[0] for n in G.nodes()]
                gy = [n[1] for n in G.nodes()]
                print(f"[INFO] 管線 X 範圍: {min(gx):.1f} ~ {max(gx):.1f}")
                print(f"[INFO] 管線 Y 範圍: {min(gy):.1f} ~ {max(gy):.1f}")
        current_data["components_info"] = components_info
        gr.Info(f"已載入 {len(components_info)} 個元件位置！")
    except Exception as e:
        import traceback
        print(f"[ERROR] 元件萃取失敗: {e}")
        traceback.print_exc()
        current_data["components_info"] = []
        gr.Warning(f"元件萃取失敗: {e}")
    
    progress(1.0, desc="繪製中...")
    return update_ui()

def update_ui(filter_block_ids=None):
    """
    根據目前的圖形分析結果 (current_data)，產生用於 Gradio 介面顯示的最新數據。

    Args:
        filter_block_ids (list, optional): 僅在圖表中顯示指定的 Block ID，None 表示全部顯示。

    Returns:
        tuple: (Plot 物件, 轉角點數量, 線段屬性 DataFrame, 元件統計 DataFrame, Block ID Dropdown 更新)。
    """
    G = current_data["graph"]
    comp_df = current_data["component_df"]
    corners = current_data["corner_points"]
    
    if G is None or comp_df is None:
        return None, 0, None, None, gr.Dropdown(choices=[], value=[])
    
    # 建立表格數據，包含座標與長度
    line_rows = []
    for u, v, d in G.edges(data=True):
        line_rows.append({
            DISP_BLOCK_ID: d.get(EDGE_BLOCK_ID, 1),
            DISP_LINE_ID: d.get(EDGE_LINE_ID, ''),
            DISP_START_X: u[0],
            DISP_START_Y: u[1],
            DISP_END_X: v[0],
            DISP_END_Y: v[1],
            DISP_LENGTH: d.get(EDGE_LENGTH, 0)
        })
    
    df_lines = pd.DataFrame(line_rows).sort_values(by=DISP_BLOCK_ID)
    
    # 取得所有 Block ID 供篩選下拉選單使用
    block_ids = sorted(df_lines[DISP_BLOCK_ID].unique().tolist())
    block_choices = [str(int(b)) for b in block_ids]
    
    # 產生含有角點標記的互動式 Plotly 圖形
    components_info = current_data.get("components_info", [])
    fig_main = visualize_graph_components(
        G,
        highlight_points=corners,
        filter_block_ids=filter_block_ids,
        components_info=components_info,
    )
    
    # 保留目前的篩選值（若有）
    current_filter_values = [str(int(b)) for b in filter_block_ids] if filter_block_ids else []
    
    return (
        fig_main,
        len(corners),
        df_lines,
        comp_df,
        gr.Dropdown(choices=block_choices, value=current_filter_values),
    )

def filter_plot(selected_blocks):
    """
    根據使用者選取的 Block ID 篩選並重新繪製管網互動圖。

    Args:
        selected_blocks (list): 使用者選取的 Block ID 字串列表。

    Returns:
        plotly.graph_objects.Figure: 篩選後的互動式管網圖表。
    """
    G = current_data["graph"]
    corners = current_data["corner_points"]
    
    if G is None:
        return None

    filter_ids = None
    if selected_blocks and len(selected_blocks) > 0:
        filter_ids = [int(b) for b in selected_blocks]
    
    components_info = current_data.get("components_info", [])
    return visualize_graph_components(G, highlight_points=corners, filter_block_ids=filter_ids, components_info=components_info)

def apply_edits(df_edited):
    """
    將使用者在互動式表格中對線段屬性（如長度或區塊 ID）的修改套用至全域圖形物件中。

    Args:
        df_edited (pd.DataFrame): 經過介面編輯後的 DataFrame。

    Returns:
        tuple: 更新後的 UI 顯示數據。
    """
    G = current_data["graph"]
    if G is None:
        return update_ui()
    
    for _, row in df_edited.iterrows():
        try:
            u = make_node(row[DISP_START_X], row[DISP_START_Y])
            v = make_node(row[DISP_END_X], row[DISP_END_Y])
            if G.has_edge(u, v):
                G[u][v][EDGE_LENGTH] = float(row[DISP_LENGTH])
                G[u][v][EDGE_BLOCK_ID] = int(row[DISP_BLOCK_ID])
        except Exception:
            continue
    
    return update_ui()

def apply_component_edits(df_comp_edited):
    """
    手動更新元件統計表的數據。

    Args:
        df_comp_edited (pd.DataFrame): 編輯後的元件統計 DataFrame。

    Returns:
        tuple: 更新後的 UI 顯示數據。
    """
    current_data["component_df"] = df_comp_edited
    return update_ui()

def handle_export():
    """
    執行報表匯出功能，產生線段清單與元件統計的 CSV 檔案供下載。

    Returns:
        tuple: (str, str) 兩個 CSV 檔案的暫存路徑。若無資料則傳回 (None, None)。
    """
    G = current_data["graph"]
    comp_df = current_data["component_df"]
    corners = current_data["corner_points"]
    
    if G is None or comp_df is None:
        gr.Warning("請先上傳檔案並完成分析後再進行匯出。")
        return None, None
    
    l_file, c_file = export_to_csv(G, comp_df, corners,download=True)
    return l_file, c_file

with gr.Blocks(title="SmartBOM Manager") as demo:
    gr.Markdown("# SmartBOM 互動式管網管理介面")

    with gr.Tabs():
        with gr.TabItem("1. 資料上傳"):
            with gr.Row():
                file_input = gr.File(label="上傳 CAD DWG 檔案")
                load_btn = gr.Button("開始分析管網與元件", variant="primary")
            gr.Markdown("請先在此分頁上傳檔案，系統將自動分析拓撲、計算元件與標註轉角點。")

        with gr.TabItem("2. 線段顯示及修改"):
            gr.Markdown("**操作提示**: 將滑鼠移到圖表上的線段即可查看詳細資訊（Line #、Block ID、長度、座標），可用滾輪縮放、拖曳平移。")
            with gr.Row():
                plot_lines = gr.Plot(label="管網佈局圖 (Hover 查看線段資訊，紅色點為轉角)", scale=3)
                with gr.Column(scale=1):
                    corner_count = gr.Number(label="偵測到的轉角點個數", precision=0)
                    block_filter = gr.Dropdown(
                        label="篩選 Block ID（留空顯示全部）",
                        choices=[],
                        value=[],
                        multiselect=True,
                        interactive=True,
                        info="選擇要顯示的管網區塊"
                    )
            
            table_lines = gr.Dataframe(
                label="線段屬性編輯器 (可直接修改 Block ID 或 Length)",
                interactive=True,
                datatype=["number", "str", "number", "number", "number", "number", "number"]
            )
            with gr.Row():
                save_lines_btn = gr.Button("儲存線段修改", variant="secondary")
                load_comp_btn = gr.Button("顯示元件位置 (需等待運算)", variant="secondary")

        with gr.TabItem("3. 元件顯示及修改"):
            table_comp = gr.Dataframe(
                label="元件 BOM 統計表",
                interactive=True
            )
            save_comp_btn = gr.Button("更新元件統計資料", variant="secondary")

        with gr.TabItem("4. 報表匯出"):
            gr.Markdown("### 點擊下方按鈕生成 CSV 報表檔")
            export_btn = gr.Button("生成並匯出報表", variant="primary")
            with gr.Row():
                out_lines_csv = gr.File(label="下載線段資料 (含座標、長度與區塊ID)")
                out_comp_csv = gr.File(label="下載元件統計 (含元件與轉角點數量)")

    # 綁定事件
    outputs = [plot_lines, corner_count, table_lines, table_comp, block_filter]
    load_btn.click(process_file, inputs=file_input, outputs=outputs)
    save_lines_btn.click(apply_edits, inputs=table_lines, outputs=outputs)
    save_comp_btn.click(apply_component_edits, inputs=table_comp, outputs=outputs)
    
    # Block ID 篩選事件（僅更新圖表）
    block_filter.change(filter_plot, inputs=block_filter, outputs=plot_lines)
    
    # 元件位置載入事件
    load_comp_btn.click(load_components, outputs=outputs)
    
    # 匯出事件
    export_btn.click(handle_export, outputs=[out_lines_csv, out_comp_csv])

if __name__ == "__main__":
    demo.launch()