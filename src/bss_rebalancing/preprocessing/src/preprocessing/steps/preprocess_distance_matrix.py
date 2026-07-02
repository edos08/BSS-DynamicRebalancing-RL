"""
Preprocess distance matrix for routing.

This module computes the shortest path distances between all pairs of nodes
in the graph.
"""

import argparse
import os

import networkx as nx
import osmnx as ox
import polars as pl

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.sources import Source, get_source
from preprocessing.core.graph import load_graph
from preprocessing.core.utils import reorder_df


def initialize_distance_matrix(graph: nx.MultiDiGraph) -> pl.DataFrame:
    """
    Initialize a distance matrix based on shortest paths in the graph.

    Parameters:
        graph: The graph representing the road network.

    Returns:
        DataFrame containing shortest path distances between all node pairs.
    """
    print("Calculating all shortest paths...")

    node_ids = ox.graph_to_gdfs(graph, edges=False).index

    # Calculate shortest paths on undirected graph
    graph_undirected = graph.to_undirected()
    distances = dict(nx.all_pairs_dijkstra_path_length(graph_undirected, weight="length"))

    matrix_data = []
    for i in node_ids:
        row = {"node_id": i}
        for j in node_ids:
            distance = distances[i].get(j, float('inf'))
            row[str(j)] = int(round(distance)) if distance != float('inf') else None
        matrix_data.append(row)

    df = pl.DataFrame(matrix_data).sort(pl.col("node_id"))
    df = reorder_df(df, 'node_id')

    return df


def run(config: PreprocessingConfig, source: Source) -> None:
    """
    Run the distance matrix preprocessing step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    print("Initializing the graph...")
    graph = load_graph(config.graph_path)

    print("Initializing the distance matrix...")
    distance_matrix = initialize_distance_matrix(graph)

    distance_matrix_path = os.path.join(config.data_path, config.distance_matrix_path)
    print(f"Saving distance matrix to {distance_matrix_path}...")
    distance_matrix.write_csv(distance_matrix_path)

    print(f"Distance matrix shape: {distance_matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess the distance matrix.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--source", type=str, required=True, help="Source id from sources.json.")
    parser.add_argument("--sources-json", type=str, default="core/sources.json")

    args = parser.parse_args()

    src = get_source(args.sources_json, args.source)
    config = PreprocessingConfig(source_id=src.id, sources_json=args.sources_json, data_path=args.data_path)
    run(config, src)


if __name__ == "__main__":
    main()
