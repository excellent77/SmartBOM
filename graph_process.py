import math
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree



def node_to_line_dist(p: tuple, line_start: tuple, line_end: tuple):
    """
    計算點到線段（Segment）的最短距離。

    Args:
        p (tuple): 目標點座標 (x, y)。
        line_start (tuple): 線段起點座標 (x, y)。
        line_end (tuple): 線段終點座標 (x, y)。

    Returns:
        float: 點到線段的歐幾里得距離。
    """
    px, py = p
    ux, uy = line_start
    vx, vy = line_end
    
    dx, dy = vx - ux, vy - uy
    if dx == 0 and dy == 0:
        dist = np.hypot(px - ux, py - uy)
    else:
        t = ((px - ux) * dx + (py - uy) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        closest_x = ux + t * dx
        closest_y = uy + t * dy
        dist = np.hypot(px - closest_x, py - closest_y)
    
    return dist

def point_to_infinite_line_dist(p: tuple, line_start: tuple, line_end: tuple):
    """
    計算點到無限延伸直線（Infinite Line）的垂直距離。

    Args:
        p (tuple): 目標點座標 (x, y)。
        line_start (tuple): 直線上的一點 (通常為線段起點)。
        line_end (tuple): 直線上的另一點 (通常為線段終點)。

    Returns:
        float: 點到直線的垂直距離。
    """
    px, py = p
    ux, uy = line_start
    vx, vy = line_end
    
    dx, dy = vx - ux, vy - uy
    if dx == 0 and dy == 0:
        return np.hypot(px - ux, py - uy)
    
    dist = abs(dy * px - dx * py + vx * uy - vy * ux) / np.hypot(dx, dy)
    
    return dist



class Graph_Data(object):
    """
    處理管網數據並建立拓撲圖的類別。
    """

    def __init__(self, pipes_df:pd.DataFrame, texts_df:pd.DataFrame=None):
        """
        初始化 Graph_Data 物件。

        Args:
            pipes_df (pd.DataFrame): 包含管線幾何資訊的 DataFrame。
            texts_df (pd.DataFrame): 包含文字標註資訊的 DataFrame。
        """
        self.pipes_data = pipes_df
        self.texts_data = texts_df
        self.graph = nx.Graph()

    def check_flange(self, flange_data:list, tolerance:float=10):
        """
        檢查並將法蘭節點整合至圖中。

        Args:
            flange_data (list): 法蘭座標列表。
            tolerance (float, optional): 距離容許誤差。預設為 10。

        Returns:
            None
        """
        nodes = list(self.graph.nodes())
        nodes_tree = cKDTree(nodes)
        for x, y in flange_data:
            flange = (x, y)
            length, node_idx = nodes_tree.query(flange, k=1)
            if length < tolerance:
                self.graph.add_edge(nodes[node_idx], flange, length=0, type='flange')

    def check_T_connection(self, tolerance=1e-2):
        """
        檢查並建立 T 型連接（三通）。
        若節點位於其他管線上，則建立虛擬連接邊。

        Args:
            tolerance (float, optional): 距離容許誤差。預設為 1e-2。

        Returns:
            None
        """
        edges = [edge for edge in self.graph.edges(data=True) if edge[2]!=0]
        for edge in edges:
            s1, e1, _ = edge
            for edge2 in edges:
                s2, e2, _ = edge2
                if edge == edge2: continue
                if self.graph.degree(s1) == 1 and node_to_line_dist(s1, s2, e2) < tolerance:
                    if self.graph.degree(s2) != 1:
                        self.graph.add_edge(s1, s2, length=0, type='virtual_connection')
                    elif self.graph.degree(e2) != 1:
                        self.graph.add_edge(s1, e2, length=0, type='virtual_connection')
                    else:
                        self.graph.add_edge(s1, s2, length=0, type='virtual_connection')

                if self.graph.degree(e1) == 1 and node_to_line_dist(e1, s2, e2) < tolerance:
                    if self.graph.degree(s2) != 1:
                        self.graph.add_edge(e1, s2, length=0, type='virtual_connection')
                    elif self.graph.degree(e2) != 1:
                        self.graph.add_edge(e1, e2, length=0, type='virtual_connection')
                    else:
                        self.graph.add_edge(e1, s2, length=0, type='virtual_connection')
    
    def check_collinear_extension(self, dist_tolerance=100, tolerance=1e-2):
        """
        檢查並延伸共線的懸空端點，修復斷開的管線連接。

        Args:
            dist_tolerance (float, optional): 雙點間可接合的容許距離。預設為 100。
            tolerance (float, optional): 共線判定的容許誤差。預設為 1e-2。

        Returns:
            None
        """
        while True:
            dangling_nodes = [node for node, degree in self.graph.degree() if degree == 1]
            if not dangling_nodes:
                break

            new_connections_found = False
            
            for u in dangling_nodes:
                if self.graph.degree(u) != 1: continue # 在迴圈中可能已經被連接
                v = list(self.graph.neighbors(u))[0]

                min_dist = float('inf')
                closest_node = None

                for w in dangling_nodes:
                    if w == u or w == v: continue

                    if point_to_infinite_line_dist(w, u, v) < tolerance:
                        dot_product = (w[0] - u[0]) * (u[0] - v[0]) + (w[1] - u[1]) * (u[1] - v[1])
                        if dot_product > 0:
                            dist_u_w = np.hypot(w[0] - u[0], w[1] - u[1])
                            if dist_u_w < min_dist:
                                min_dist = dist_u_w
                                closest_node = w
                
                if closest_node and min_dist<dist_tolerance and not self.graph.has_edge(u, closest_node):
                    self.graph.add_edge(u, closest_node, length=0, type='virtual_extension')
                    new_connections_found = True
            
            if not new_connections_found:
                break
    
    def build_graph(self, slope_tolerance:float=0.9, dist_tolerance:float=10):
        """
        根據輸入的管線數據與文字標註，建立管網拓撲圖。
        計算文字標註與管線的關聯，以賦予管線長度屬性。

        Args:
            slope_tolerance (float, optional): 斜率相同判定的容許誤差。預設為 0.98。
            dist_tolerance (float, optional): 文字到線之間可接合的容許距離。預設為 5。

        Returns:
            nx.Graph: 建立完成的 NetworkX 圖物件。
        """
        pipe_lengths = {}
        if self.texts_data is not None:
            print("正在配對管線長度...")
            for _, text_row in self.texts_data.iterrows():
                tx, ty = text_row['位置 X'], text_row['位置 Y']
                
                angle_rad = math.radians(text_row['旋轉'])
                
                tv = np.array([math.cos(angle_rad), math.sin(angle_rad)])
                
                best_dist = float('inf')
                best_comp_idx = -1
                
                for idx, row in self.pipes_data.iterrows():
                    u = (row['起點 X'], row['起點 Y'])
                    v = (row['終點 X'], row['終點 Y'])
                    dx, dy = v[0] - u[0], v[1] - u[1]
                    length = math.hypot(dx, dy)
                    
                    ev = np.array([dx / length, dy / length])
                    
                    if abs(np.dot(tv, ev)) > slope_tolerance:
                        dist = node_to_line_dist((tx, ty), u, v)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_comp_idx = idx
                
                if best_comp_idx != -1 and best_dist < dist_tolerance:
                    try:
                        val = float(text_row['值'])
                        pipe_lengths[best_comp_idx] = pipe_lengths.get(best_comp_idx, 0) + val
                    except (ValueError, TypeError, KeyError):
                        pass
        
        for idx, row in self.pipes_data.iterrows():
            p1 = (row['起點 X'], row['起點 Y'])
            p2 = (row['終點 X'], row['終點 Y'])
            self.graph.add_edge(p1, p2, length=pipe_lengths.get(idx, 0), line_id=idx)

        return self.graph
