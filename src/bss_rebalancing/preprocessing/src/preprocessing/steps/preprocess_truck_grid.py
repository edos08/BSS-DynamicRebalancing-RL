"""
Preprocess truck grid for rebalancing operations.

This module divides the graph into cells for truck routing and assigns
nodes to cells.
"""

import argparse
import os
import pickle

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.sources import Source, get_source
from preprocessing.core.graph import load_graph
from preprocessing.core.grid import (
    assign_nodes_to_cells,
    divide_graph_into_cells,
    remove_empty_cells,
    set_adjacent_cells,
)
from preprocessing.core.plotting import plot_graph_with_grid


def run(config: PreprocessingConfig, source: Source) -> None:
    """
    Run the truck grid preprocessing step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    print("Initializing the graph...")
    graph = load_graph(config.graph_path)

    print("Dividing the graph into cells...")
    cell_dict = divide_graph_into_cells(graph, config.cell_size)

    print("Assigning nodes to cells...")
    assign_nodes_to_cells(graph, cell_dict)

    print("Removing empty cells...")
    cell_dict = remove_empty_cells(cell_dict)

    print("Setting up cell properties...")
    to_remove = []
    for key, cell in cell_dict.items():
        try:
            cell.set_center_node(graph)
        except Exception as e:
            print(f"WARNING: {e}")
            to_remove.append(key)

    for key in to_remove:
        cell_dict.pop(key)

    print("Setting adjacent cells...")
    set_adjacent_cells(cell_dict)

    # Save cell dictionary
    cell_data_path = os.path.join(config.data_path, config.cell_data_path)
    print(f"Saving cell data to {cell_data_path}...")
    with open(cell_data_path, "wb") as file:
        pickle.dump(cell_dict, file)

    print(f"Created {len(cell_dict)} cells.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess the truck grid data.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--source", type=str, required=True, help="Source id from sources.json.")
    parser.add_argument("--sources-json", type=str, default="core/sources.json")
    parser.add_argument("--cell-size", type=int, default=None, help="Cell size in meters (defaults to source's cell_size).")
    parser.add_argument("--plot", action="store_true", help="Plot the grid after processing.")
    parser.add_argument("--plot-path", type=str, default=None, help="Path to save the plot figure.")

    args = parser.parse_args()

    src = get_source(args.sources_json, args.source)
    config = PreprocessingConfig(
        source_id=src.id,
        sources_json=args.sources_json,
        data_path=args.data_path,
        cell_size=args.cell_size if args.cell_size is not None else src.cell_size,
    )
    run(config, src)


if __name__ == "__main__":
    main()
