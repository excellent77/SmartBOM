import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random
import math



TOLERANCE = 0.1



def visualize_graph_components(G, components):
    """
    使用 Matplotlib 將圖的連通分群視覺化。
    不同的顏色代表不同的獨立管網系統。
    'nut' 節點會以不同顏色標示。
    """
    plt.figure(figsize=(16, 9))
    # 節點的位置就是其自身的座標，但Y軸需要翻轉以符合螢幕座標系
    pos = {node: (node[0], -node[1]) for node in G.nodes()}
    
    print(f"將 {len(components)} 個分群繪製到圖表上...")
    # 為每個連通分群分配一個隨機顏色
    for i, component in enumerate(components):
        color = (random.random(), random.random(), random.random())
        subgraph = G.subgraph(component)
        nx.draw_networkx(subgraph, pos=pos, with_labels=False, node_color=color, node_size=15, edge_color=color, width=1.5)
        
        # 顯示邊的長度標籤 (只顯示大於 0 的長度，並格式化為整數)
        edge_labels = nx.get_edge_attributes(subgraph, 'length')
        formatted_edge_labels = {k: f"{v:.0f}" for k, v in edge_labels.items() if v > 0}
        nx.draw_networkx_edge_labels(subgraph, pos=pos, edge_labels=formatted_edge_labels, font_size=8, font_color=color)

    plt.title(f"管網連通分群視覺化 (共 {len(components)} 個獨立系統)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def node_to_line_dist(p: tuple, line_start: tuple, line_end: tuple):
    px, py = p
    ux, uy = line_start
    vx, vy = line_end
    
    # 計算點到線段的距離 (Point to Segment Distance)
    dx, dy = vx - ux, vy - uy
    if dx == 0 and dy == 0:
        dist = np.hypot(px - ux, py - uy)
    else:
        t = ((px - ux) * dx + (py - uy) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t)) # 限制 t 在 [0, 1] 之間，確保只檢查線段範圍，而非無限延伸的直線
        closest_x = ux + t * dx
        closest_y = uy + t * dy
        dist = np.hypot(px - closest_x, py - closest_y)
    
    return dist

def point_to_infinite_line_dist(p: tuple, line_start: tuple, line_end: tuple):
    """計算點到無限延伸直線的距離"""
    px, py = p
    ux, uy = line_start
    vx, vy = line_end
    
    dx, dy = vx - ux, vy - uy
    if dx == 0 and dy == 0:
        # 線段起終點相同，退化為點到點的距離
        return np.hypot(px - ux, py - uy)
    
    # 使用標準的點到線距離公式: |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / sqrt((x2-x1)^2 + (y2-y1)^2)
    dist = abs(dy * px - dx * py + vx * uy - vy * ux) / np.hypot(dx, dy)
    
    return dist

def generate_iso_bom_table(csv_file_path, visualize=False):
    print(f"解析 CAD 萃取資料: {csv_file_path}")
    
    # 1. 讀取 CSV 檔案
    df = pd.read_csv(csv_file_path, engine='python')

    # 2. 萃取管線幾何資料 (Lines)
    # 確保抓取含有起終點座標的線段
    pipes_df = df[df['名稱'].isin(['線'])].copy()
    pipes_df = pipes_df[pipes_df['圖層'].isin(['ISO圖_BG'])]
    pipes_df['長度'] = pd.to_numeric(pipes_df['長度']).fillna(0)

    # 將座標欄位轉換為數值型態，無法轉換的資料會變成 NaN (Not a Number)
    coord_cols = ['起點 X', '起點 Y', '終點 X', '終點 Y']
    for col in coord_cols:
        pipes_df[col] = pd.to_numeric(pipes_df[col], errors='coerce')

    # 3. 萃取文字註解資料 (Texts) - 用於 ENT DESCRIPTION
    # 在 AutoCAD 萃取表中，文字可能存於 '值' 或 '內容' 欄位
    text_df = df[df['名稱'].isin(['文字', '多行文字'])].copy()
    # 穩健地合併 '內容' 和 '值' 欄位，優先使用 '內容'
    text_df['註解'] = text_df['值'].fillna(text_df['內容']).fillna('')
    text_df = text_df[text_df['註解'].str.strip() != '']

    # --- 邏輯修正：將註解分類為「管線名稱」與「長度」---
    # 嘗試將註解轉換為數字，若失敗則返回 NaN
    numeric_check = pd.to_numeric(text_df['註解'], errors='coerce')

    # 凡是純數字的註解 (轉換後不是 NaN)，歸類為長度
    length_texts_df = text_df[numeric_check.notna()].copy()
    length_texts_df['length'] = numeric_check[numeric_check.notna()]
    # 確保座標是數值
    length_texts_df['位置 X'] = pd.to_numeric(length_texts_df['位置 X'], errors='coerce')
    length_texts_df['位置 Y'] = pd.to_numeric(length_texts_df['位置 Y'], errors='coerce')
    length_texts_df.dropna(subset=['位置 X', '位置 Y', 'length'], inplace=True)

    # ==========================================
    # 階段一：建立 G(V,E) 拓撲圖
    # ==========================================    
    G = nx.Graph()
    pipe_lengths = {}
    print("🕸️ 正在建立管網空間拓撲圖...")
    # 遍歷所有名稱註解，讓「文字」發射投影線去尋找最平行的「管線」
    for _, text_row in length_texts_df.iterrows():
        tx, ty = text_row['位置 X'], text_row['位置 Y']
        
        # AutoCAD 的旋轉角度通常為度數 (0度為水平向右)，轉為弧度
        angle_rad = math.radians(text_row['旋轉'])
        
        # 計算文字的方向向量 (Text Vector)
        tv = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        best_dist = float('inf')
        best_comp_idx = -1
        
        # 尋找圖中所有的管線段
        for idx, row in pipes_df.iterrows():
            # 取得起終點座標 (四捨五入至小數點後1位，容許微小的繪圖誤差)
            u = (row['起點 X'], row['起點 Y'])
            v = (row['終點 X'], row['終點 Y'])
            dx, dy = v[0] - u[0], v[1] - u[1]
            length = math.hypot(dx, dy)
            
            # 計算線段的方向向量 (Edge Vector)
            ev = np.array([dx / length, dy / length])
            
            # 檢查平行度：計算兩個向量的內積絕對值
            # 若內積絕對值大於 0.98 (允許約 11 度的繪圖誤差)，則視為平行
            if abs(np.dot(tv, ev)) > 0.98:
                # 若平行，計算點到線段的垂直投影距離
                dist = node_to_line_dist((tx, ty), u, v)
                
                # 更新距離該文字最近的線段
                if dist < best_dist:
                    best_dist = dist
                    best_comp_idx = idx
        
        # 將找到的標籤賦予該線段所屬的管網分群
        if best_comp_idx != -1 and best_dist < 10:
           pipe_lengths[best_comp_idx] = pipe_lengths.get(best_comp_idx, 0) + text_row['length']

    for idx, row in pipes_df.iterrows():
        # 取得起終點座標 (四捨五入至小數點後1位，容許微小的繪圖誤差)
        p1 = (row['起點 X'], row['起點 Y'])
        p2 = (row['終點 X'], row['終點 Y'])
        G.add_edge(p1, p2, length=pipe_lengths.get(idx, 0), line_id=idx)

    #處理法蘭定位
    NUT_TOLERANCE = 10
    nut_texts_df = df[df['名稱'].isin(['GAS-ISO-F-NUT'])].copy()
    nodes = list(G.nodes())
    nodes_tree = cKDTree(nodes)
    
    for x, y in nut_texts_df[['位置 X', '位置 Y']].values:
        nut = (x, y)
        length, node_idx = nodes_tree.query(nut, k=1)
        if length < NUT_TOLERANCE:
            G.add_edge(nodes[node_idx], nut, length=0, type='nut')


    # --- 處理 T 型連接與線段重疊 ---
    # 檢查是否有節點位於其他線段上 (非端點對端點，而是端點在線段中間)
    
    edges = [edge for edge in G.edges(data=True) if edge[2]!=0]

    for edge in edges:
        s1, e1, _ = edge
        for edge2 in edges:
            s2, e2, _ = edge2
            if edge == edge2: continue
            if G.degree(s1) == 1 and node_to_line_dist(s1, s2, e2) < TOLERANCE:
                if G.degree(s2) != 1:
                    G.add_edge(s1, s2, length=0, type='virtual_connection')
                elif G.degree(e2) != 1:
                    G.add_edge(s1, e2, length=0, type='virtual_connection')
                else:
                    G.add_edge(s1, s2, length=0, type='virtual_connection')

            if G.degree(e1) == 1 and node_to_line_dist(e1, s2, e2) < TOLERANCE:
                if G.degree(s2) != 1:
                    G.add_edge(e1, s2, length=0, type='virtual_connection')
                elif G.degree(e2) != 1:
                    G.add_edge(e1, e2, length=0, type='virtual_connection')
                else:
                    G.add_edge(e1, s2, length=0, type='virtual_connection')
    
    # 階段 2.2: 處理共線但不相連的線段 (Collinear Extension)
    print("... 正在延伸懸空端點以修正連線...")
    TOLERANCE_COLLINEAR = 1e-2
    
    # 反覆執行直到沒有新的連線產生，以處理鏈式連接的情況
    
    while True:
        dangling_nodes = [node for node, degree in G.degree() if degree == 1]
        if not dangling_nodes:
            break

        new_connections_found = False
        
        for u in dangling_nodes:
            if G.degree(u) != 1: continue # 在迴圈中可能已經被連接
            v = list(G.neighbors(u))[0]

            min_dist = float('inf')
            closest_node = None

            for w in dangling_nodes:
                if w == u or w == v: continue

                if point_to_infinite_line_dist(w, u, v) < TOLERANCE_COLLINEAR:
                    dot_product = (w[0] - u[0]) * (u[0] - v[0]) + (w[1] - u[1]) * (u[1] - v[1])
                    if dot_product > 0:
                        dist_u_w = np.hypot(w[0] - u[0], w[1] - u[1])
                        if dist_u_w < min_dist:
                            min_dist = dist_u_w
                            closest_node = w
            
            if closest_node and min_dist<100 and not G.has_edge(u, closest_node):
                G.add_edge(u, closest_node, length=0, type='virtual_extension')
                new_connections_found = True
        
        if not new_connections_found:
            break
    # ==========================================
    # 階段二：利用空間索引 (KD-Tree) 綁定註解與管網
    # ==========================================
    # 取得所有的管網獨立群組 (Connected Components)
    connected_components = list(nx.connected_components(G))
    print(f"✅ 發現 {len(connected_components)} 個獨立的管網連通分群。")

    bom_records = []
    for comp_idx, component in enumerate(connected_components):
        label = f"區塊 {comp_idx + 1}"
        subgraph = G.subgraph(component)
        comp_length = sum([d['length'] for u, v, d in subgraph.edges(data=True)])
        bom_records.append({
            "ENT DESCRIPTION": label,
            "length": comp_length
        })

    # ==========================================
    # 階段三：資料聚合與產出 PDF 格式報表
    # ==========================================
    # 相同規格的管網可能被中斷，這裡根據 ENT DESCRIPTION 進行群組加總
    bom_df = pd.DataFrame(bom_records)
    if not bom_df.empty:
        # 依照長度排序 (選用)
        summary_df = bom_df.sort_values('length', ascending=False)
    else:
        summary_df = pd.DataFrame(columns=['ENT DESCRIPTION', 'length'])

    # 格式化為附圖的表格結構 (NO, ENT DESCRIPTION, QTY)
    final_table = pd.DataFrame({
        'NO': range(1, len(summary_df) + 1),
        'ENT DESCRIPTION': summary_df['ENT DESCRIPTION'],
        'QTY': summary_df['length'].apply(lambda x: f"{int(x)}mm") # 轉為整數並加上 mm
    })

    print("\n📊 --- 最終自動化 BOM 表產出 (匹配出圖檔格式) ---")
    print(final_table.to_markdown(index=False))

    if visualize:
        print("🎨 正在產生視覺化圖表...")
        visualize_graph_components(G, connected_components)
    
    return final_table



# 執行函式
if __name__ == "__main__":
    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"
    generate_iso_bom_table(file_name, visualize=True)
