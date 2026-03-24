import pandas as pd
import networkx as nx

from graph_process import Graph_Data
from units_process import calculate_line, generate_graph_report, generate_component_report, visualize_graph_components




if __name__ == "__main__":

    file_name = "/home/excellent/SmartBOM/data/單線圖_N_BGAS_LP.csv"
    "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"

    df = pd.read_csv(file_name, engine='python')

    graph = calculate_line(df)
    connected_components = list(nx.connected_components(graph))
    generate_graph_report(graph, connected_components)
    generate_component_report(df, ["Reducer", "Hose", "Ball Valve", "Check Valve (CV)", "Diaphragm Valve", "Plate", "VCR"])
    visualize_graph_components(graph, connected_components)
    
            