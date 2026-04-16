import pandas as pd
import networkx as nx

from units_process import find_corner_points, calculate_line, generate_graph_report, generate_component_report, visualize_graph_components




if __name__ == "__main__":

    file_name = "/home/excellent/SmartBOM/data/ISO圖_N_BGAS_5001_SPTS_V2.csv"
    "/home/excellent/SmartBOM/data/單線圖_N_BGAS_LP-1.csv"

    df = pd.read_csv(file_name, engine='python')

    graph = calculate_line(df)
    corner_points = find_corner_points(graph)
    generate_graph_report(graph)
    print(f"轉角點數量: {len(corner_points)}")
    generate_component_report(df, ["Reducer", "Hose", "Ball Valve", "Check Valve (CV)", "Diaphragm Valve", "Plate", "VCR"])
    visualize_graph_components(graph, corner_points)
    
            