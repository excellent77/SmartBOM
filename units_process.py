import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from dxf_reader import read_dwg
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
    
    # 檢查並篩選存在於 DataFrame 中的欄位，若不存在則印出警告並繼續執行
    valid_cols = []
    for col in columns_name:
        if col in processed_df.columns:
            valid_cols.append(col)
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")

    processed_df = processed_df[valid_cols]
    
    def convert_safe(x):
        try:
            return pd.to_numeric(x)
        except (ValueError, TypeError):
            return x

    for col in valid_cols:
        processed_df[col] = processed_df[col].apply(convert_safe)
    
    processed_df.dropna(subset=valid_cols, inplace=True)

    return processed_df

def check_block(df:pd.DataFrame, name:str):
    """
    檢查資料中是否包含特定名稱的圖塊 (Block)，並統計其數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str): 欲搜尋的圖塊名稱關鍵字（不分大小寫）。

    Returns:
        float or bool: 若找到對應圖塊則回傳數量總和，否則回傳 False。
    """
    processed_df = df.copy()
    categories = processed_df['名稱'].unique()
    for value in categories:
        if name in str(value).lower():
            processed_df['計數'] = pd.to_numeric(processed_df['計數'], errors='coerce')
            return processed_df[processed_df['名稱']==value]['計數'].sum()
    return False

def calculate_ball_value(df:pd.DataFrame, name:str='ball_value', color:str="顏色_181"):
    """
    計算球閥 (Ball Valve) 的數量。
    邏輯：優先查找圖塊名稱，若無則篩選指定顏色中的圓形或橢圓形。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'ball_value'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_181"。

    Returns:
        int: 計算出的數量。
    """
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
    """
    計算隔膜閥 (Diaphragm Valve) 的數量。
    邏輯：透過顏色過濾線段，建立圖形結構後計算連通分群。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'dia valve'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_3"。

    Returns:
        int: 計算出的數量。
    """
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
    
def calculate_CV(df:pd.DataFrame, name:str='cv', color:str="顏色_212"):
    """
    計算止回閥 (Check Valve) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'cv'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_212"。

    Returns:
        int: 計算出的數量。
    """
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
    
def calculate_Regulagtor(df:pd.DataFrame, name:str='regulator', color:str="顏色_6"):
    """
    計算壓力調節閥 (Regulator) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'regulator'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_6"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['弧'])]
        return df['計數'].sum()
    
def calculate_3P_Regulagtor(df:pd.DataFrame, name:str='3p_regulator', color:str="顏色_140"):
    """
    計算 3P 調節閥 (3P Regulator) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 '3p_regulator'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_140"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['弧'])]
        return df['計數'].sum()
    
def calculate_Tee(df:pd.DataFrame, name:str='Tee', color:str="顏色_134"):
    """
    計算三通 (Tee) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Tee'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_134"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    return value if value else process_df_data(
        df,
        columns_name=['計數'],
        conditions=[('出圖型式', [color])]
    )['計數'].sum()

def calculate_reducer(df:pd.DataFrame, name:str='reducer', color:str="顏色_5"):
    """
    計算大小頭 (Reducer) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'reducer'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_5"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    return value if value else process_df_data(
        df,
        columns_name=['計數'],
        conditions=[('出圖型式', [color])]
    )['計數'].sum()

def calculate_ELBOW(df:pd.DataFrame, name:str='elbow', color:str="顏色_16"):
    """
    計算彎頭 (Elbow) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'elbow'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_16"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    return value if value else process_df_data(
        df,
        columns_name=['計數'],
        conditions=[('出圖型式', [color])]
    )['計數'].sum()

def calculate_NUT_F(df:pd.DataFrame, name:str='NUT(F)', color:str="顏色_1"):
    """
    計算螺帽(母) (NUT F) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'NUT(F)'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_1"。

    Returns:
        int: 計算出的數量。
    """
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
    
def calculate_NUT_M(df:pd.DataFrame, name:str='NUT(M)', color:str="顏色_8"):
    """
    計算螺帽(公) (NUT M) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'NUT(M)'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_8"。

    Returns:
        int: 計算出的數量。
    """
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
    
def calculate_S_Gland(df:pd.DataFrame, name:str='S Gland', color:str="顏色_142"):
    """
    計算 S Gland 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'S Gland'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_142"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()
    
def calculate_L_Gland(df:pd.DataFrame, name:str='L Gland', color:str="顏色_11"):
    """
    計算 L Gland 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'L Gland'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_11"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()
    
def calculate_GASKET(df:pd.DataFrame, name:str='GASKET', color:str="顏色_165"):
    """
    計算墊片 (Gasket) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'GASKET'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_165"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()
    
def calculate_VCR_Tee(df:pd.DataFrame, name:str='VCR Tee', color:str="顏色_7"):
    """
    計算 VCR 三通的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'VCR Tee'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_7"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['填充線'])]
        return df['計數'].sum()

def calculate_Union_R_Tee(df:pd.DataFrame, name:str='Union R.Tee', color:str="顏色_171"):
    """
    計算 Union R.Tee 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Union R.Tee'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_171"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['填充線'])]
        return df['計數'].sum()
    
def calculate_Union_Tee(df:pd.DataFrame, name:str='Union Tee', color:str="顏色_241"):
    """
    計算 Union Tee 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Union Tee'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_241"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()
    
def calculate_Gauge(df:pd.DataFrame, name:str='Gauge', color:str="顏色_40"):
    """
    計算壓力表 (Gauge) 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Gauge'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_40"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['圓'])]
        return df['計數'].sum()

def calculate_Union(df:pd.DataFrame, name:str='Union', color:str="顏色_67"):
    """
    計算 Union 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Union'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_67"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()

def calculate_Reducer_Union(df:pd.DataFrame, name:str='Reducer Union', color:str="顏色_211"):
    """
    計算 Reducer Union 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'Reducer Union'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_211"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=['計數', '名稱'],
            conditions=[('出圖型式', [color])]
        )
        df = df[df['名稱'].isin(['聚合線'])]
        return df['計數'].sum()

def calculate_hose(df:pd.DataFrame, name:str='hose', color:str="顏色_4"):
    """
    計算軟管 (Hose) 的數量。
    邏輯：透過顏色過濾弧線中心點，並利用空間聚類 (cKDTree) 辨識同一元件。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'hose'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_4"。

    Returns:
        int: 計算出的數量。
    """
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

'''def calculate_Gauge(df:pd.DataFrame, name:str='Gauge', color:str="顏色_40"):
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

        return len(components)'''

def calculate_line(df:pd.DataFrame, color:str="顏色_30"):
    """
    建立管網拓撲圖，包含管線幾何處理、長度標註配對與連接修復。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        color (str, optional): 代表管線的顏色標籤。預設為 "顏色_30"。

    Returns:
        nx.Graph: 構建完成並經過連通性修復後的 NetworkX 圖物件。
    """
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

    Returns:
        None
    """
    connected_components = list(nx.connected_components(graph))
    print(f"✅ 發現 {len(connected_components)} 個獨立的管網連通分群。")

    bom_records = []
    for comp_idx, component in enumerate(connected_components):
        label = f"區塊 {comp_idx + 1}"
        subgraph = graph.subgraph(component)
        comp_length = sum([d.get('length', 0) for u, v, d in subgraph.edges(data=True)])
        bom_records.append({
            "ENT DESCRIPTION": label,
            "length": comp_length
        })

    bom_df = pd.DataFrame(bom_records, columns=["ENT DESCRIPTION", "length"])
    if bom_df.empty:
        print("⚠️ 沒有可生成的管網報表，graph 可能為空或未包含任何邊。")
        return

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

    Returns:
        matplotlib.figure.Figure: 繪製完成的管網視覺化圖表物件。
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
        "Ball Valve": calculate_ball_value,
        "Diaphragm Valve": calculate_dia_valve,
        "Check Valve (CV)": calculate_CV,
        "Regulagtor": calculate_Regulagtor,
        "3P Regulagtor": calculate_3P_Regulagtor,
        "Tee": calculate_Tee,
        "Reducer": calculate_reducer,
        "ELBOW": calculate_ELBOW,
        "NUT(F)": calculate_NUT_F,
        "NUT(M)": calculate_NUT_M,
        "S Gland": calculate_S_Gland,
        "L Gland": calculate_L_Gland,
        "GASKET": calculate_GASKET,
        "VCR Tee": calculate_VCR_Tee,
        "Union R.Tee": calculate_Union_R_Tee,
        "Union Tee": calculate_Union_Tee,
        "Gauge": calculate_Gauge,
        "Union": calculate_Union,
        "Reducer Union": calculate_Reducer_Union,
        "Hose": calculate_hose,
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

def export_to_csv(
        graph: nx.Graph,
        component_df: pd.DataFrame,
        corner_points: list,
        line_path: str = "lines_report.csv",
        component_path: str = "components_report.csv",
        download:bool = True
    ):
    """
    將管線資料與元件統計匯出為兩個 CSV 檔案。

    Args:
        graph (nx.Graph): 拓撲圖物件。
        component_df (pd.DataFrame): 元件統計 DataFrame。
        corner_points (list): 轉角點座標列表。
        line_path (str): 線段 CSV 儲存路徑。
        component_path (str): 元件 CSV 儲存路徑。

    Returns:
        tuple: (str, str) 分別為線段報表與元件統計報表的存檔路徑。
    """
    # 1. 產出線段報表
    line_data = []
    for u, v, d in graph.edges(data=True):
        line_data.append({
            "block_id": d.get('block_id', 1),
            "start_x": u[0],
            "start_y": u[1],
            "end_x": v[0],
            "end_y": v[1],
            "length": d.get('length', 0)
        })
    df_lines = pd.DataFrame(line_data)
    if download:
        df_lines.to_csv(line_path, index=False, encoding='utf-8-sig')

    # 2. 產出元件報表 (包含轉角點數量)
    df_comp_final = component_df.copy()
    corner_row = pd.DataFrame([{"Component": "Corner Points", "QTY": len(corner_points)}])
    df_comp_final = pd.concat([df_comp_final, corner_row], ignore_index=True)
    if download:
        df_comp_final.to_csv(component_path, index=False, encoding='utf-8-sig')

    return line_path, component_path

if __name__ == "__main__":
    '''# 設定輸入檔案與基礎路徑
    file_name = "/home/li-cho-yueh/SmartBOM/data/16_單線圖_KOXDLP2000.dwg"
    # 讀取 CAD DWG 資料
    df = read_dwg(file_name)
    # 建立管網拓撲並修復連通性
    graph = calculate_line(df)
    # 生成管網連通報告
    generate_graph_report(graph)
    # 找出系統中的轉角點
    corner_points = find_corner_points(graph)
    # 執行所有元件的 BOM 統計
    report_df = generate_component_report(df, report_items= [
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
    ])
    # 視覺化管網結果
    visualize_graph_components(graph)
    # 匯出分析結果至 CSV 檔案
    #export_to_csv(graph, report_df, corner_points)'''

    import os
    data_dir = "/home/li-cho-yueh/SmartBOM/data/"
    os.makedirs(os.path.join(data_dir, "reports"), exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith(".dwg"):
            print(f"Generated report: {file}")
            df = read_dwg(os.path.join(data_dir, file))
            # 建立管網拓撲並修復連通性
            graph = calculate_line(df)
            # 生成管網連通報告
            generate_graph_report(graph)
            # 找出系統中的轉角點
            corner_points = find_corner_points(graph)
            # 執行所有元件的 BOM 統計
            report_df = generate_component_report(df, report_items= [
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
            ])
            # 匯出分析結果至 CSV 檔案
            export_to_csv(
                graph,
                report_df,
                corner_points,
                line_path=os.path.join(data_dir, "reports", f"lines_{file}.csv"),
                component_path=os.path.join(data_dir, "reports", f"components_{file}.csv")
            )