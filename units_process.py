import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import cKDTree

from dxf_reader import read_dwg
from graph_process import Graph_Data
from columns import (
    COL_NAME, COL_COLOR, COL_COUNT,
    COL_START_X, COL_START_Y, COL_END_X, COL_END_Y,
    COL_CENTER_X, COL_CENTER_Y,
    COL_POS_X, COL_POS_Y, COL_VALUE, COL_ROTATION,
    ETYPE_LINE, ETYPE_POLYLINE, ETYPE_CIRCLE, ETYPE_ELLIPSE,
    ETYPE_ARC, ETYPE_TEXT, ETYPE_MTEXT, ETYPE_HATCH,
    EDGE_LENGTH, EDGE_LINE_ID, EDGE_BLOCK_ID,
    DISP_BLOCK_ID, DISP_LINE_ID, DISP_START_X, DISP_START_Y,
    DISP_END_X, DISP_END_Y, DISP_LENGTH,
    DISP_COMPONENT, DISP_QTY,
)



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
    categories = processed_df[COL_NAME].unique()
    for value in categories:
        if name in str(value).lower():
            processed_df[COL_COUNT] = pd.to_numeric(processed_df[COL_COUNT], errors='coerce')
            return processed_df[processed_df[COL_NAME]==value][COL_COUNT].sum()
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_CIRCLE, ETYPE_ELLIPSE])]
        return df[COL_COUNT].sum()

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
            columns_name=[COL_START_X, COL_START_Y, COL_END_X, COL_END_Y],
            conditions=[(COL_NAME, [ETYPE_LINE]), (COL_COLOR, [color])]
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_CIRCLE, ETYPE_ELLIPSE])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_ARC])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_ARC])]
        return df[COL_COUNT].sum()
    
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
        columns_name=[COL_COUNT],
        conditions=[(COL_COLOR, [color])]
    )[COL_COUNT].sum()

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
        columns_name=[COL_COUNT],
        conditions=[(COL_COLOR, [color])]
    )[COL_COUNT].sum()

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
        columns_name=[COL_COUNT],
        conditions=[(COL_COLOR, [color])]
    )[COL_COUNT].sum()

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
            columns_name=[COL_START_X, COL_START_Y, COL_END_X, COL_END_Y],
            conditions=[(COL_NAME, [ETYPE_LINE]), (COL_COLOR, [color])]
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
            columns_name=[COL_START_X, COL_START_Y, COL_END_X, COL_END_Y],
            conditions=[(COL_NAME, [ETYPE_LINE]), (COL_COLOR, [color])]
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_HATCH])]
        return df[COL_COUNT].sum()
    
def calculate_VCR_U_Tee(df:pd.DataFrame, name:str='VCR 正 Tee', color:str="顏色_84"):
    """
    計算 VCR 正 Tee 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'VCR 正 Tee'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_84"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()

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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_HATCH])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()
    
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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_CIRCLE])]
        return df[COL_COUNT].sum()

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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()

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
            columns_name=[COL_COUNT, COL_NAME],
            conditions=[(COL_COLOR, [color])]
        )
        df = df[df[COL_NAME].isin([ETYPE_POLYLINE])]
        return df[COL_COUNT].sum()

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
        columns_name=[COL_CENTER_X, COL_CENTER_Y],
        conditions=[(COL_COLOR, [color]), (COL_NAME, [ETYPE_ARC])]
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

def calculate_CAP(df:pd.DataFrame, name:str='CAP', color:str="顏色_37"):
    """
    計算 CAP 的數量。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。
        name (str, optional): 圖塊名稱。預設為 'CAP'。
        color (str, optional): 辨識用的顏色標籤。預設為 "顏色_37"。

    Returns:
        int: 計算出的數量。
    """
    value = check_block(df, name)
    if value:
        return value
    else:
        df = process_df_data(
            df,
            columns_name=[COL_START_X, COL_START_Y, COL_END_X, COL_END_Y],
            conditions=[(COL_NAME, [ETYPE_LINE]), (COL_COLOR, [color])]
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

        return len(components)/2
    
    
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
        columns_name=[COL_START_X, COL_START_Y, COL_END_X, COL_END_Y],
        conditions=[(COL_NAME, [ETYPE_LINE, ETYPE_HATCH, ETYPE_POLYLINE]), (COL_COLOR, [color])]
    )
    lengths_df = process_df_data(
        df,
        columns_name=[COL_POS_X, COL_POS_Y, COL_VALUE, COL_ROTATION],
        conditions=[(COL_NAME, [ETYPE_TEXT, ETYPE_MTEXT])]
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
        comp_length = sum([d.get(EDGE_LENGTH, 0) for u, v, d in subgraph.edges(data=True)])
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


def visualize_graph_components(G:nx.Graph, highlight_points:list=None, filter_block_ids:list=None, components_info:list=None):
    """
    使用 Plotly 視覺化管網的連通分群，支援互動式 Hover 與篩選。

    Args:
        G (nx.Graph): NetworkX 圖物件。
        highlight_points (list, optional): 需要特別標示的點座標列表 [(x, y), ...]。
        filter_block_ids (list, optional): 僅顯示指定 Block ID 的線段，None 表示全部顯示。
        components_info (list, optional): 元件資訊列表 [(x, y, name), ...]。

    Returns:
        plotly.graph_objects.Figure: 繪製完成的管網互動式視覺化圖表物件。
    """
    fig = go.Figure()

    # 固定色盤，確保每次渲染顏色一致
    COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    ]

    # 優先檢查邊屬性中的 block_id，若無則使用連通分群作為預設分群方式
    edge_blocks = nx.get_edge_attributes(G, EDGE_BLOCK_ID)

    if edge_blocks:
        blocks = {}
        for (u, v), b_id in edge_blocks.items():
            blocks.setdefault(b_id, []).append((u, v))
    else:
        components = list(nx.connected_components(G))
        blocks = {i+1: list(G.subgraph(c).edges()) for i, c in enumerate(components)}

    # 套用 Block ID 篩選
    if filter_block_ids and len(filter_block_ids) > 0:
        blocks = {k: v for k, v in blocks.items() if k in filter_block_ids}

    print(f"將 {len(blocks)} 個區塊繪製到圖表上...")

    global_line_idx = 0
    for block_id in sorted(blocks.keys()):
        edges = blocks[block_id]
        color = COLORS[(block_id - 1) % len(COLORS)]

        # 收集此 block 中所有線段的座標與 hover 資訊
        x_vals, y_vals = [], []
        customdata_list = []

        for u, v in edges:
            line_id = G[u][v].get(EDGE_LINE_ID, '')
            length = G[u][v].get(EDGE_LENGTH, 0)

            x_vals.extend([u[0], v[0], None])
            y_vals.extend([u[1], v[1], None])

            # 為起點與終點都附上相同的線段資訊，None 分隔點用空字串
            info = [
                str(global_line_idx),   # 0: Line #
                str(block_id),          # 1: Block ID
                str(line_id),           # 2: Line ID
                f"{length:.1f}",        # 3: Length
                f"{u[0]:.2f}",          # 4: Start X
                f"{u[1]:.2f}",          # 5: Start Y
                f"{v[0]:.2f}",          # 6: End X
                f"{v[1]:.2f}",          # 7: End Y
            ]
            customdata_list.extend([info, info, ['']*8])
            global_line_idx += 1

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=5, color=color),
            name=f"Block {block_id}",
            legendgroup=f"block_{block_id}",
            customdata=customdata_list,
            hovertemplate=(
                "<b>Line #%{customdata[0]}</b><br>"
                "Block ID: %{customdata[1]}<br>"
                "Line ID: %{customdata[2]}<br>"
                "長度: %{customdata[3]} mm<br>"
                "起點: (%{customdata[4]}, %{customdata[5]})<br>"
                "終點: (%{customdata[6]}, %{customdata[7]})"
                "<extra></extra>"
            ),
        ))

        # 在每條線段中點標註長度
        for u, v in edges:
            length = G[u][v].get(EDGE_LENGTH, 0)
            if length > 0:
                mid_x = (u[0] + v[0]) / 2
                mid_y = (u[1] + v[1]) / 2
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"{length:.0f}",
                    showarrow=False,
                    font=dict(size=9, color=color),
                    bgcolor="rgba(255,255,255,0.75)",
                    borderpad=1,
                )

    # 繪製高亮點 (例如角點)
    if highlight_points:
        hx = [p[0] for p in highlight_points]
        hy = [p[1] for p in highlight_points]
        fig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle',
                        line=dict(color='black', width=1.5)),
            name='轉角點 (Corner)',
            hovertemplate="<b>轉角點</b><br>座標: (%{x:.2f}, %{y:.2f})<extra></extra>",
        ))

    # 繪製元件圖示
    if components_info:
        print(f"[INFO] 正在繪製 {len(components_info)} 個元件圖示到圖表上")
        
        # 1. 根據元件名稱分組
        grouped_comps = {}
        for cx, cy, cname in components_info:
            grouped_comps.setdefault(cname, {"x": [], "y": []})
            grouped_comps[cname]["x"].append(cx)
            grouped_comps[cname]["y"].append(cy)
            
        # 2. 定義變化用的圖示與顏色集
        symbols = ['diamond', 'square', 'circle', 'triangle-up', 'triangle-down', 'cross', 'x', 'pentagon', 'hexagram', 'star']
        colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000'
        ]
        
        # 3. 為每種元件建立獨立的 trace (模組化，可在圖例單獨開關)
        for i, (cname, coords) in enumerate(grouped_comps.items()):
            fig.add_trace(go.Scatter(
                x=coords["x"], 
                y=coords["y"],
                mode='markers',  # 移除文字標籤避免太雜亂，改用 hover 顯示
                marker=dict(
                    size=12,  # 尺寸縮小 (原本22)
                    color=colors[i % len(colors)],
                    symbol=symbols[i % len(symbols)],
                    line=dict(color='black', width=1)
                ),
                name=f'{cname} ({len(coords["x"])})',  # 圖例上顯示數量
                customdata=[cname]*len(coords["x"]),
                hovertemplate="<b>%{customdata}</b><br>座標: (%{x:.2f}, %{y:.2f})<extra></extra>",
            ))

    fig.update_layout(
        title=f"管網佈局圖 (共 {len(blocks)} 個區塊)",
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y"),
        hovermode='closest',
        showlegend=True,
        legend=dict(title="區塊 (Block)", x=1.02, y=1, bordercolor="grey", borderwidth=1),
        height=700,
        template='plotly_white',
        dragmode='pan',
    )

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
        "VCR 正 Tee": calculate_VCR_U_Tee,
        "Union R.Tee": calculate_Union_R_Tee,
        "Union Tee": calculate_Union_Tee,
        "Gauge": calculate_Gauge,
        "Union": calculate_Union,
        "Reducer Union": calculate_Reducer_Union,
        "Hose": calculate_hose,
        "CAP(母塞頭)": calculate_CAP,
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
            results.append({DISP_COMPONENT: name, DISP_QTY: int(qty)})
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
            DISP_BLOCK_ID: d.get(EDGE_BLOCK_ID, 1),
            DISP_START_X: u[0],
            DISP_START_Y: u[1],
            DISP_END_X: v[0],
            DISP_END_Y: v[1],
            DISP_LENGTH: d.get(EDGE_LENGTH, 0)
        })
    df_lines = pd.DataFrame(line_data)
    if download:
        df_lines.to_csv(line_path, index=False, encoding='utf-8-sig')

    # 2. 產出元件報表 (包含轉角點數量)
    df_comp_final = component_df.copy()
    corner_row = pd.DataFrame([{DISP_COMPONENT: "Corner Points", DISP_QTY: len(corner_points)}])
    df_comp_final = pd.concat([df_comp_final, corner_row], ignore_index=True)
    if download:
        df_comp_final.to_csv(component_path, index=False, encoding='utf-8-sig')

    if download:
        return line_path, component_path
    else:
        return df_lines, df_comp_final

def build_reports(
        file_path:str,
        download:bool=True,
        line_path=f"lines.csv",
        component_path=f"components.csv",        
    )->tuple[pd.DataFrame, pd.DataFrame]:
    """
    將管線資料與元件統計匯出為兩個 CSV 檔案。

    Args:
        graph (nx.Graph): 拓撲圖物件。
        component_df (pd.DataFrame): 元件統計 DataFrame。
        corner_points (list): 轉角點座標列表。
        line_path (str): 線段 CSV 儲存路徑。
        component_path (str): 元件 CSV 儲存路徑。

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame) 分別為線段報表與元件統計報表之檔案
    """
    df = read_dwg(file_path)
    graph = calculate_line(df)
    generate_graph_report(graph)
    corner_points = find_corner_points(graph)
    component_df = generate_component_report(df, report_items= [
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
        "VCR 正 Tee",
        "Union R.Tee",
        "Union Tee",
        "Gauge",
        "Union",
        "Reducer Union",
        "Hose",
        "CAP(母塞頭)"
    ])

    line_df, component_df = export_to_csv(
        graph, component_df, corner_points,
        download=download,
        line_path=line_path,
        component_path=component_path
    )
    return line_df, component_df




if __name__ == "__main__":
    # 設定輸入檔案與基礎路徑
    #file_name = "/home/f11167/SmartBOM/data/1_單線圖_KMACLM0100.dxf"
    #build_reports(file_name) #輸出兩個檔案之DF

    idx = 0
    for path in os.listdir("/home/f11167/SmartBOM/data/data_new/"):
        if path.endswith(".dwg"):
            idx += 1
            print(f"Processing file: {path}")
            build_reports(
                f"/home/f11167/SmartBOM/data/data_new/{path}",
                download=True,
                line_path=f"/home/f11167/SmartBOM/data/reports/{path[:-4]}_lines.csv",
                component_path=f"/home/f11167/SmartBOM/data/reports/{path[:-4]}_components.csv"
            )

    '''path = "/home/f11167/SmartBOM/data/16_單線圖_KOXDLP2000.dxf"
    build_reports(
        path,
        download=True,
        line_path=f"/home/f11167/SmartBOM/lines.csv",
        component_path=f"/home/f11167/SmartBOM/components.csv"
    )'''