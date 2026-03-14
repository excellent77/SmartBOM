import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def process_df_data(
        df:pd.DataFrame,
        columns_name:list,
        conditions:list
        ):
    """
    根據指定條件過濾 DataFrame 並轉換欄位型態。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        columns_name (list): 需要保留並轉換為數值的欄位名稱列表。
        conditions (list): 篩選條件列表，格式為 [(欄位名, [關鍵字列表]), ...]。

    Returns:
        pd.DataFrame: 處理後的 DataFrame，僅包含數值化的指定欄位且移除 NaN 值。
    """
    processed_df = df.copy()
    for name, key_word in conditions:
        processed_df = processed_df[processed_df[name].isin(key_word)]
    
    processed_df = processed_df[columns_name]
    
    for col in columns_name:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    processed_df.dropna(subset=columns_name, inplace=True)
    return processed_df

def generate_graph_report(graph:nx.Graph, connected_components:list):
    """
    生成管網連通分群的 BOM 報告並輸出至控制台。

    Args:
        graph (nx.Graph): NetworkX 圖物件，包含管網拓撲與邊屬性。
        connected_components (list): 連通分群列表，每個元素為一組節點集合。
    """
    print(f"✅ 發現 {len(connected_components)} 個獨立的管網連通分群。")

    bom_records = []
    for comp_idx, component in enumerate(connected_components):
        label = f"區塊 {comp_idx + 1}"
        subgraph = graph.subgraph(component)
        comp_length = sum([d['length'] for u, v, d in subgraph.edges(data=True)])
        bom_records.append({
            "ENT DESCRIPTION": label,
            "length": comp_length
        })

    bom_df = pd.DataFrame(bom_records)
    summary_df = bom_df.sort_values('length', ascending=False)

    final_table = pd.DataFrame({
        'NO': range(1, len(summary_df) + 1),
        'ENT DESCRIPTION': summary_df['ENT DESCRIPTION'],
        'QTY': summary_df['length'].apply(lambda x: f"{int(x)}mm")
    })

    print("\n--- 最終自動化 BOM 表產出 ---")
    print(final_table.to_markdown(index=False))


def visualize_graph_components(G:nx.Graph, components:list):
    """
    使用 Matplotlib 視覺化管網的連通分群。

    Args:
        G (nx.Graph): NetworkX 圖物件。
        components (list): 連通分群列表，用於區分不同顏色的子圖。
    """
    plt.figure(figsize=(16, 9))
    pos = {node: (node[0], -node[1]) for node in G.nodes()}
    
    print(f"將 {len(components)} 個分群繪製到圖表上...")
    for i, component in enumerate(components):
        color = (random.random(), random.random(), random.random())
        subgraph = G.subgraph(component)
        nx.draw_networkx(subgraph, pos=pos, with_labels=False, node_color=[color], node_size=15, edge_color=color, width=1.5)
        
        edge_labels = nx.get_edge_attributes(subgraph, 'length')
        formatted_edge_labels = {k: f"{v:.0f}" for k, v in edge_labels.items() if v > 0}
        nx.draw_networkx_edge_labels(subgraph, pos=pos, edge_labels=formatted_edge_labels, font_size=8, font_color=color)

    plt.title(f"Visualized Connected Components({len(components)} independent systems)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"
    df = pd.read_csv(file_name, engine='python')
    '''data = process_df_data(
        df,
        columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
        conditions=[('名稱', ['線']), ('圖層', ['ISO圖_BG'])]
    )'''
    data = process_df_data(
        df,
        columns_name=['位置 X', '位置 Y', '值'],
        conditions=[('名稱', ['文字', '多行文字'])]
    )
    '''data = process_df_data(
        df,
        columns_name=['位置 X', '位置 Y'],
        conditions=[('名稱', ['GAS-ISO-F-NUT'])]
    )'''
    print(data)