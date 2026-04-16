import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from graph_process import Graph_Data



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
        if processed_df.get(name) is None:
            raise ValueError(f"Column '{name}' not found in the DataFrame.")
        processed_df = processed_df[processed_df[name].isin(key_word)]
    
    processed_df = processed_df[columns_name]
    
    def convert_safe(x):
        try:
            return pd.to_numeric(x)
        except (ValueError, TypeError):
            return x

    for col in columns_name:
        processed_df[col] = processed_df[col].apply(convert_safe)
    
    processed_df.dropna(subset=columns_name, inplace=True)

    return processed_df

def check_block(df:pd.DataFrame, name:str):
    processed_df = df.copy()
    categories = processed_df['名稱'].unique()
    for value in categories:
        if name in str(value).lower():
            processed_df['計數'] = pd.to_numeric(processed_df['計數'], errors='coerce')
            return processed_df[processed_df['名稱']==value]['計數'].sum()
    return False


def calculate_reducer(df:pd.DataFrame, name:str='reducer', color:str="顏色_5"):
    value = check_block(df, name)
    return value if value else process_df_data(
        df,
        columns_name=['計數'],
        conditions=[('出圖型式', [color])]
    )['計數'].sum()

def calculate_hose(df:pd.DataFrame, name:str='hose', color:str="顏色_4"):
    value = check_block(df, name)
    if value:
        return value
    
    hose_df = process_df_data(
        df,
        columns_name=['中心點 X', '中心點 Y'],
        conditions=[('出圖型式', [color]), ('名稱', ['弧'])]
    )

    if hose_df.empty:
        return 0

    points = list(zip(hose_df['中心點 X'], hose_df['中心點 Y']))
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=5)
    
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    graph.add_edges_from(pairs)
    
    return nx.number_connected_components(graph)

def calculate_ball_value(df:pd.DataFrame, name:str='ball_value', color:str="顏色_181"):
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['圓', '橢圓'])]
        return df['計數'].sum()

def calculate_CV(df:pd.DataFrame, name:str='cv', color:str="顏色_6"):
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['圓', '橢圓'])]
        return df['計數'].sum()

def calculate_dia_valve(df:pd.DataFrame, name:str='dia valve', color:str="顏色_3"):
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
            conditions=[('名稱', ['線']), ('出圖型式', [color])]
        )
        graph = Graph_Data(df)
        graph.build_graph()
        connected_components = list(nx.connected_components(graph.graph))
        components = []
        for component in connected_components:
            subgraph = graph.graph.subgraph(component)
            if len(components) > 0 and len(subgraph.nodes) < len(components[-1]):
                continue
            while len(components) > 0 and len(subgraph.nodes) > len(components[-1]):
                components.pop()
            components.append(subgraph.nodes)
            
        return len(components)

def calculate_plate(df:pd.DataFrame, name:str='plate', color:str="顏色_40"):
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
            conditions=[('名稱', ['線']), ('出圖型式', [color])]
        )
        graph = Graph_Data(df)
        graph.build_graph()
        connected_components = list(nx.connected_components(graph.graph))
        components = []
        for component in connected_components:
            subgraph = graph.graph.subgraph(component)
            if len(components) > 0 and len(subgraph.nodes) < len(components[-1]):
                continue
            while len(components) > 0 and len(subgraph.nodes) > len(components[-1]):
                components.pop()
            components.append(subgraph.nodes)

        return len(components)

def calculate_VCR(df:pd.DataFrame, name:str='vcr', color:str="顏色_1"):
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
            conditions=[('名稱', ['線']), ('出圖型式', [color])]
        )
        graph = Graph_Data(df)
        graph.build_graph()
        connected_components = list(nx.connected_components(graph.graph))
        components = []
        for component in connected_components:
            subgraph = graph.graph.subgraph(component)
            if len(components) > 0 and len(subgraph.nodes) < len(components[-1]):
                continue
            while len(components) > 0 and len(subgraph.nodes) > len(components[-1]):
                components.pop()
            components.append(subgraph.nodes)

        return len(components)

def calculate_line(df:pd.DataFrame, color:str="顏色_30"):
    lines_df = process_df_data(
        df,
        columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
        conditions=[('名稱', ['線', '填充線', '聚合線']), ('出圖型式', [color])]
    )
    lengths_df = process_df_data(
        df,
        columns_name=['位置 X', '位置 Y', '值', '旋轉'],
        conditions=[('名稱', ['文字', '多行文字'])]
    )
    graph = Graph_Data(lines_df, lengths_df)
    graph.build_graph()
    graph.check_T_connection()
    graph.check_collinear_extension()
    return graph.graph

def find_corner_points(graph: nx.Graph, slope_tolerance: float = 0.98):
        """
        找出管網中的所有轉角點（角點）。
        邏輯：若一個節點連接兩條線（Degree 為 2），且這兩條線不平行（共線），則該點為轉角點。

        Args:
            slope_tolerance (float, optional): 判定平行的斜率容許值（dot product 絕對值）。
                                             值越接近 1 代表判定平行越嚴格。預設為 0.98。

        Returns:
            list: 轉角點的座標列表 [(x, y), ...]。
        """
        corners = []
        for node, degree in graph.degree():
            if degree >= 2:
                neighbors = list(graph.neighbors(node))
                permutations = []
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        permutations.append((i, j))

                for i, j in permutations:
                    v1, v2 = neighbors[i], neighbors[j]
                    
                    # 計算從該節點出發的兩個向量
                    vec1 = np.array([v1[0] - node[0], v1[1] - node[1]])
                    vec2 = np.array([v2[0] - node[0], v2[1] - node[1]])
                    
                    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                    if n1 > 0 and n2 > 0:
                        # 計算正規化點積的絕對值，若小於容許值則視為轉角
                        dot_product_abs = abs(np.dot(vec1 / n1, vec2 / n2))
                        if dot_product_abs < slope_tolerance:
                            corners.append(node)
                            break
        return corners

def generate_graph_report(graph:nx.Graph):
    """
    生成管網連通分群的 BOM 報告並輸出至控制台。

    Args:
        graph (nx.Graph): NetworkX 圖物件，包含管網拓撲與邊屬性。
        connected_components (list): 連通分群列表，每個元素為一組節點集合。
    """
    connected_components = list(nx.connected_components(graph))
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


def visualize_graph_components(G:nx.Graph, highlight_points:list=None):
    """
    使用 Matplotlib 視覺化管網的連通分群。

    Args:
        G (nx.Graph): NetworkX 圖物件。
        highlight_points (list, optional): 需要特別標示的點座標列表 [(x, y), ...]。
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    pos = {node: (node[0], node[1]) for node in G.nodes()}

    # 優先檢查邊屬性中的 block_id，若無則使用連通分群作為預設分群方式
    edge_blocks = nx.get_edge_attributes(G, 'block_id')
    
    if edge_blocks:
        # 根據 block_id 將邊分組
        blocks = {}
        for (u, v), b_id in edge_blocks.items():
            blocks.setdefault(b_id, []).append((u, v))
        components_edges = list(blocks.values())
    else:
        components = list(nx.connected_components(G))
        components_edges = [list(G.subgraph(c).edges()) for c in components]

    print(f"將 {len(components_edges)} 個區塊繪製到圖表上...")
    for edges in components_edges:
        color = (random.random(), random.random(), random.random())
        nodes = set([n for e in edges for n in e])
        
        # 繪製節點與邊，明確指定繪圖軸為 ax
        nx.draw_networkx_nodes(G, pos=pos, nodelist=list(nodes), node_color=[color], node_size=15, ax=ax)
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color=color, width=1.5, ax=ax)

        # 標註線段長度標籤
        edge_labels = { (u, v): f"{G[u][v].get('length', 0):.0f}" for u, v in edges if G[u][v].get('length', 0) > 0 }
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=8, font_color=color, ax=ax)

    # 繪製高亮點 (例如角點)
    if highlight_points:
        hx = [p[0] for p in highlight_points]
        hy = [p[1] for p in highlight_points]
        ax.scatter(hx, hy, color='red', s=50, marker='o', label='Highlighted Points', edgecolors='black', zorder=10)
        ax.legend()

    ax.set_title(f"Visualized Systems (Total Blocks: {len(components_edges)})")
    ax.set_aspect('equal')
    ax.grid(True)
    return fig

def generate_component_report(df: pd.DataFrame, report_items: list):
    """
    整合各個元件計算函式，生成並列印元件統計報表。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        report_items (list): 包含元件名稱與計算函式的列表。

    Returns:
        pd.DataFrame: 包含元件名稱與數量的報表 DataFrame。
    """

    report_dict = {
        "Reducer": calculate_reducer,
        "Hose": calculate_hose,
        "Ball Valve": calculate_ball_value,
        "Check Valve (CV)": calculate_CV,
        "Diaphragm Valve": calculate_dia_valve,
        "Plate": calculate_plate,
        "VCR": calculate_VCR,
    }

    results = []
    print("正在統計元件數量...")
    for name in report_items:
        if report_dict.get(name, False):
            func = report_dict[name]
        else:
            raise ValueError(f"找不到 {name}")

        try:
            qty = func(df)
            results.append({"Component": name, "QTY": int(qty)})
        except Exception as e:
            print(f"計算 {name} 時發生錯誤: {e}")

    report_df = pd.DataFrame(results)
    if not report_df.empty:
        print("\n--- Component Report ---")
        print(report_df.to_markdown(index=False))
    else:
        print("未發現任何指定元件。")
    
    return report_df


if __name__ == "__main__":
    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"
    df = pd.read_csv(file_name, engine='python')
    graph = calculate_line(df, color='ByLayer')
    generate_graph_report(graph)
    generate_component_report(df, ["Reducer", "Hose", "Ball Valve", "Check Valve (CV)", "Diaphragm Valve", "Plate", "VCR"])
    visualize_graph_components(graph)