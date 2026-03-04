import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random



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

def check_on_line(p: tuple, line_start: tuple, line_end: tuple):
    px, py = p
    ux, uy = line_start
    vx, vy = line_end
    # y = ax +b
    a = (vy - uy)/(vx - ux) if vx != ux else float('inf')
    if a==float('inf'):
        return px == ux
    else:
        b = uy - a * ux
        return py == (a*px + b)

    


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

    # 移除任何含有無效 (NaN) 座標的管線資料，確保所有節點都是有限數值
    #pipes_df.dropna(subset=coord_cols, inplace=True)

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

    # 其他的註解 (轉換後是 NaN)，歸類為管線名稱
    name_texts_df = text_df[numeric_check.isna()].copy()
    name_texts_df['ENT DESCRIPTION'] = name_texts_df['註解'].str.strip()
    # 確保座標是數值
    name_texts_df['位置 X'] = pd.to_numeric(name_texts_df['位置 X'], errors='coerce')
    name_texts_df['位置 Y'] = pd.to_numeric(name_texts_df['位置 Y'], errors='coerce')
    name_texts_df.dropna(subset=['位置 X', '位置 Y'], inplace=True)

    # 移除可能只包含空白的名稱
    name_texts_df = name_texts_df[name_texts_df['ENT DESCRIPTION'] != '']

    # ==========================================
    # 階段一：建立 G(V,E) 拓撲圖
    # ==========================================
    length_coords = length_texts_df[['位置 X', '位置 Y']].values
    length_tree = cKDTree(length_coords)

    nut_texts_df = name_texts_df[name_texts_df['ENT DESCRIPTION'] == 'GAS-ISO-F-NUT']
    nut_coords = nut_texts_df[['位置 X', '位置 Y']].values
    nut_tree = cKDTree(nut_coords)
    is_nut = lambda node: nut_tree.query(node)[0] < 1

    G = nx.Graph()
    print("🕸️ 正在建立管網空間拓撲圖...")
    for idx, row in pipes_df.iterrows():
        # 取得起終點座標 (四捨五入至小數點後1位，容許微小的繪圖誤差)
        p1 = (round(row['起點 X']), round(row['起點 Y']))
        p2 = (round(row['終點 X']), round(row['終點 Y']))
        length_dist, length_idx = length_tree.query(np.array([p1, p2]).mean(axis=0))
        if length_dist < 100:
            length = length_texts_df.iloc[length_idx]['length']
            #print(length)
        else:
            print('fail')
        
        # 建立圖的邊 (Edge)，並帶入長度屬性
        G.add_edge(p1, p2, length=length, line_id=idx)

    # --- 處理 T 型連接與線段重疊 ---
    # 檢查是否有節點位於其他線段上 (非端點對端點，而是端點在線段中間)
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    
    for edge in edges:
        s1, e1, _ = edge
        s1_is_connected, e1_is_connected = is_nut(s1), is_nut(e1)
        min_dist_s1, min_node_s1 = None, None
        min_dist_e1, min_node_e1 = None, None
        
        for edge2 in edges:
            if edge == edge2:
                continue
            s2, e2, _ = edge2
            if s1 == s2 or s1 == e2:
                s1_is_connected = True
                continue
            if e1 == s2 or e1 == e2:
                e1_is_connected = True
                continue
            
            if s1_is_connected and e1_is_connected:
                break

            if not s1_is_connected:
                if node_to_line_dist(s1, s2, e2) < 1:
                    G.add_edge(s1, s2, length=0, type='virtual_connection')
                    s1_is_connected = True

                if check_on_line(s1, s2, e2):
                    dist = np.hypot(s1[0] - s2[0], s1[1] - s2[1])
                    if min_dist_s1 is None or dist < min_dist_s1:
                        min_dist_s1, min_node_s1 = dist, s2
                    dist = np.hypot(s1[0] - e2[0], s1[1] - e2[1])
                    if min_dist_s1 is None or dist < min_dist_s1:
                        min_dist_s1, min_node_s1 = dist, e2


            if not e1_is_connected:
                if node_to_line_dist(e1, s2, e2) < 1:
                    G.add_edge(e1, s2, length=0, type='virtual_connection')
                    e1_is_connected = True

                if check_on_line(e1, s2, e2):
                    dist = np.hypot(e1[0] - s2[0], e1[1] - s2[1])
                    if min_dist_e1 is None or dist < min_dist_e1:
                        min_dist_e1, min_node_e1 = dist, s2
                    dist = np.hypot(e1[0] - e2[0], e1[1] - e2[1])
                    if min_dist_e1 is None or dist < min_dist_e1:
                        min_dist_e1, min_node_e1 = dist, e2

        if not s1_is_connected and min_node_s1 is not None:
            G.add_edge(s1, min_node_s1, length=0, type='virtual_connection')

        if not e1_is_connected and min_node_e1 is not None:
            G.add_edge(e1, min_node_e1, length=0, type='virtual_connection')
            

    # 取得所有的管網獨立群組 (Connected Components)
    connected_components = list(nx.connected_components(G))
    print(f"✅ 發現 {len(connected_components)} 個獨立的管網連通分群。")

    if visualize:
        print("🎨 正在產生視覺化圖表...")
        visualize_graph_components(G, connected_components)

    # ==========================================
    # 階段二：利用空間索引 (KD-Tree) 綁定註解與管網
    # ==========================================
    bom_records = []
    
    # 若圖面中有成功分類的名稱與長度註解，則進行空間匹配
    
    print(f"🔎 發現 {len(name_texts_df)} 個名稱註解與 {len(length_texts_df)} 個長度註解，開始進行匹配...")
    # 1. 分別為「名稱」與「長度」註解建立 KD-Tree
    name_coords = name_texts_df[['位置 X', '位置 Y']].values
    name_tree = cKDTree(name_coords)

    for comp_idx, component in enumerate(connected_components):
        # --- 修正開始 ---
        # a. 計算管網群組的幾何中心點 (Centroid)
        comp_nodes = np.array(list(component))
        centroid = comp_nodes.mean(axis=0)

        # b. 分別查詢最近的「名稱」和「長度」
        name_dist, name_idx = name_tree.query(centroid)
        # c. 檢查兩者是否都在合理範圍內
        if name_dist < 1000:
            # 成功匹配，從各自的 DataFrame 中獲取資訊
            label = name_texts_df.iloc[name_idx]['ENT DESCRIPTION']
        else:
            # 匹配失敗，使用備用邏輯
            print('fail')
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
        #summary_df = bom_df.groupby('ENT DESCRIPTION', as_index=False)['length'].sum()
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
    
    return final_table

def visualize_graph_components(G, components):
    """
    使用 Matplotlib 將圖的連通分群視覺化。
    不同的顏色代表不同的獨立管網系統。
    """
    plt.figure(figsize=(16, 9))
    # 節點的位置就是其自身的座標，但Y軸需要翻轉以符合螢幕座標系
    pos = {node: (node[0], -node[1]) for node in G.nodes()}
    
    print(f"將 {len(components)} 個分群繪製到圖表上...")
    # 為每個連通分群分配一個隨機顏色
    for i, component in enumerate(components):
        color = (random.random(), random.random(), random.random())
        subgraph = G.subgraph(component)
        nx.draw_networkx(subgraph, pos=pos, with_labels=False, node_color=[color], node_size=15, edge_color=color, width=1.5)

    plt.title(f"管網連通分群視覺化 (共 {len(components)} 個獨立系統)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# 執行函式
if __name__ == "__main__":
    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"
    generate_iso_bom_table(file_name, visualize=True)