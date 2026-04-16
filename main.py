import pandas as pd
import gradio as gr
import networkx as nx
from units_process import calculate_line, visualize_graph_components, find_corner_points, generate_component_report

# 全域變數用於存儲目前處理中的圖形數據
current_data = {
    "graph": None, 
    "df": None, 
    "corner_points": [],
    "component_df": None
}

COMPONENT_LIST = ["Reducer", "Hose", "Ball Valve", "Check Valve (CV)", "Diaphragm Valve", "Plate", "VCR"]

def process_file(file):
    if file is None:
        return None, None, None, None
    df = pd.read_csv(file.name, engine='python')
    current_data["df"] = df
    graph = calculate_line(df)
    
    # 初始化：將連通分群設定為初始區塊 ID (Block ID)
    components = list(nx.connected_components(graph))
    for i, nodes in enumerate(components):
        sub = graph.subgraph(nodes)
        for u, v in sub.edges():
            graph[u][v]['block_id'] = i + 1
            
    current_data["graph"] = graph
    corners = find_corner_points(graph)
    current_data["corner_points"] = corners
    current_data["component_df"] = generate_component_report(df, COMPONENT_LIST)
    
    gr.Info("資料讀取與管網分析成功！")
    return update_ui()

def update_ui():
    G = current_data["graph"]
    comp_df = current_data["component_df"]
    corners = current_data["corner_points"]
    
    if G is None or comp_df is None:
        return None, 0, None, None
    
    # 建立表格數據，包含座標與長度
    line_rows = []
    for u, v, d in G.edges(data=True):
        line_rows.append({
            "Block ID": d.get('block_id', 1),
            "Line ID": d.get('line_id', ''),
            "Length (mm)": d.get('length', 0)
        })
    
    df_lines = pd.DataFrame(line_rows).sort_values(by="Block ID")
    
    # 產生含有角點標記的圖形作為主要顯示
    fig_main = visualize_graph_components(G, highlight_points=corners)
    
    return fig_main, len(corners), df_lines, comp_df

def apply_edits(df_edited):
    G = current_data["graph"]
    if G is None:
        return update_ui()
    
    for _, row in df_edited.iterrows():
        try:
            u = eval(row["Start Point"])
            v = eval(row["End Point"])
            if G.has_edge(u, v):
                G[u][v]['length'] = float(row["Length (mm)"])
                G[u][v]['block_id'] = int(row["Block ID"])
        except:
            continue
    
    return update_ui()

def apply_component_edits(df_comp_edited):
    current_data["component_df"] = df_comp_edited
    return update_ui()

with gr.Blocks(title="SmartBOM Manager") as demo:
    gr.Markdown("# SmartBOM 互動式管網管理介面")

    with gr.Tabs():
        with gr.TabItem("1. 資料上傳"):
            with gr.Row():
                file_input = gr.File(label="上傳 CAD CSV 資料")
                load_btn = gr.Button("開始分析管網與元件", variant="primary")
            gr.Markdown("請先在此分頁上傳檔案，系統將自動分析拓撲、計算元件與標註轉角點。")

        with gr.TabItem("2. 線段顯示及修改"):
            with gr.Row():
                plot_lines = gr.Plot(label="管網佈局圖 (紅色點為轉角)", scale=3)
                corner_count = gr.Number(label="偵測到的轉角點個數", precision=0)
            
            table_lines = gr.Dataframe(
                label="線段屬性編輯器 (可直接修改 Block ID 或 Length)",
                interactive=True,
                datatype=["number", "str", "str", "str", "number"]
            )
            save_lines_btn = gr.Button("儲存線段修改", variant="secondary")

        with gr.TabItem("3. 元件顯示及修改"):
            table_comp = gr.Dataframe(
                label="元件 BOM 統計表",
                interactive=True
            )
            save_comp_btn = gr.Button("更新元件統計資料", variant="secondary")

    # 綁定事件
    outputs = [plot_lines, corner_count, table_lines, table_comp]
    load_btn.click(process_file, inputs=file_input, outputs=outputs)
    save_lines_btn.click(apply_edits, inputs=table_lines, outputs=outputs)
    save_comp_btn.click(apply_component_edits, inputs=table_comp, outputs=outputs)

if __name__ == "__main__":
    demo.launch()