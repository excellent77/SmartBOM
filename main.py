import pandas as pd
import networkx as nx
import data_process, graph_process



if __name__ == "__main__":

    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"

    df = pd.read_csv(file_name, engine='python')

    pipes_data = data_process.process_df_data(
        df,
        columns_name=['起點 X', '起點 Y', '終點 X', '終點 Y'],
        conditions=[('名稱', ['線']), ('圖層', ['ISO圖_BG'])]
    )
    lengths_data = data_process.process_df_data(
        df,
        columns_name=['位置 X', '位置 Y', '值', '旋轉'],
        conditions=[('名稱', ['文字', '多行文字'])]
    )
    flange_data = data_process.process_df_data(
        df,
        columns_name=['位置 X', '位置 Y'],
        conditions=[('名稱', ['GAS-ISO-F-NUT'])]
    )
    flange_data = flange_data.values.tolist()

    graph_data = graph_process.Graph_Data(pipes_data, lengths_data)
    graph_data.build_graph()
    graph_data.check_flange(flange_data)
    graph_data.check_T_connection()
    graph_data.check_collinear_extension()

    connected_components = list(nx.connected_components(graph_data.graph))
    data_process.generate_graph_report(graph_data.graph, connected_components)
    data_process.visualize_graph_components(graph_data.graph, connected_components)
    
    