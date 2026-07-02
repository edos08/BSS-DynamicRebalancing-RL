"""
Interpolate rate data to build PMF matrices.

This module interpolates the sparse rate matrices to create probability
mass function (PMF) matrices for trip distribution.
"""

import argparse
import os
import math
from typing import Dict, Tuple

import numpy as np
import osmnx as ox
import polars as pl
from haversine import haversine_vector, Unit
from tqdm import tqdm

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.sources import Source, get_source
from preprocessing.core.graph import load_graph
from preprocessing.core.utils import nodes_within_radius, reorder_df


def interpolate_row(
        row_data: np.ndarray,
        data_cols: list,
        nearby_nodes_dict: Dict[int, Dict[int, Tuple[float, float]]],
        nodes_dict: Dict[int, Tuple[float, float]],
        col_to_idx: Dict[int, int]
) -> np.ndarray:
    """
    Interpolate zero values in a single row using IDW based on destination geography.

    Parameters:
        row_data: 1D numpy array representing a single row of the matrix
        data_cols: List of column names (node IDs as strings)
        nearby_nodes_dict: Dictionary mapping node_id -> {nearby_node_id: (lat, lon)}
        nodes_dict: Dictionary mapping node_id -> (lat, lon)
        col_to_idx: Dictionary mapping column node_id -> column index

    Returns:
        Interpolated row data as numpy array
    """
    # Work on a copy to avoid modifying original
    interpolated_row = row_data.copy()

    zero_mask = interpolated_row == 0
    non_zero_mask = ~zero_mask

    if not zero_mask.any():
        return interpolated_row

    zero_col_indices = np.where(zero_mask)[0]
    non_zero_col_indices = np.where(non_zero_mask)[0]
    non_zero_col_ids = {int(data_cols[i]) for i in non_zero_col_indices}

    # Vectorized interpolation for all zero columns
    for col_idx in zero_col_indices:
        col_node_id = int(data_cols[col_idx])
        nearby_nodes = nearby_nodes_dict.get(col_node_id, {})

        # Find nearby nodes that have non-zero values
        nearby_non_zero = {
            nz_id: nearby_nodes[nz_id]
            for nz_id in non_zero_col_ids
            if nz_id in nearby_nodes
        }

        if nearby_non_zero:
            coords = nodes_dict[col_node_id]

            # Vectorized distance computation
            nearby_items = list(nearby_non_zero.items())
            nearby_node_ids = [item[0] for item in nearby_items]
            nearby_coords = np.array([item[1] for item in nearby_items])

            dists = haversine_vector(
                [coords] * len(nearby_coords),
                nearby_coords,
                unit=Unit.METERS
            )

            # Avoid division by zero for exact matches
            dists = np.maximum(dists, 1e-10)

            # Get rates for nearby nodes
            nearby_indices = np.array([col_to_idx[nz_id] for nz_id in nearby_node_ids])
            rates = interpolated_row[nearby_indices]

            # IDW interpolation
            weights = 1.0 / dists
            interpolated_row[col_idx] = np.sum(rates * weights) / np.sum(weights)

    return interpolated_row


def build_pmf_matrix(
    rate_matrix: pl.DataFrame,
    nearby_nodes_dict: Dict[int, Dict[int, Tuple[float, float]]],
    nodes_dict: Dict[int, Tuple[float, float]]
) -> pl.DataFrame:
    """
    Build the PMF matrix from the rate matrix using spatial interpolation.

    Parameters:
        rate_matrix: The rate matrix (sparse) as Polars DataFrame.
        nearby_nodes_dict: Dictionary of nearby nodes for each node.
        nodes_dict: Dictionary of node coordinates.

    Returns:
        Interpolated PMF matrix as Polars DataFrame.
    """
    # Store original column order and node_id values for reconstruction
    original_node_ids = rate_matrix['node_id'].to_list()
    original_data_cols = rate_matrix.select(pl.exclude('node_id')).columns

    # Filter out the exclude_node from both rows and columns
    exclude_node = 10000
    exclude_node_str = str(exclude_node)

    # Store the original row and column for node 10000 (if they exist)
    has_exclude_row = exclude_node in original_node_ids
    has_exclude_col = exclude_node_str in original_data_cols

    if has_exclude_row:
        exclude_row_original = rate_matrix.filter(pl.col('node_id') == exclude_node)
    else:
        exclude_row_original = None

    if has_exclude_col:
        # Store the entire column including the exclude_node row
        exclude_col_original = rate_matrix.select(['node_id', exclude_node_str])
    else:
        exclude_col_original = None

    # Remove row with node_id == exclude_node
    filtered_df = rate_matrix.filter(pl.col('node_id') != exclude_node)

    # Remove column with name == exclude_node (if it exists)
    data_cols = filtered_df.select(pl.exclude(['node_id', exclude_node_str])).columns

    # Extract node IDs from the index column
    node_ids_index = filtered_df['node_id'].to_list()

    # Extract data as numpy array
    pmf_array = filtered_df.select(data_cols).to_numpy().astype(np.float64)

    # Create mappings: node_id -> array index
    col_to_idx = {int(col): idx for idx, col in enumerate(data_cols)}
    row_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids_index)}

    # Identify non-zero rows
    row_sums = pmf_array.sum(axis=1)
    non_zero_row_mask = row_sums != 0
    non_zero_row_indices = np.where(non_zero_row_mask)[0]

    # Stage 1: Interpolate zero cells in non-zero rows
    for row_idx in non_zero_row_indices:
        row_data = pmf_array[row_idx, :]
        pmf_array[row_idx, :] = interpolate_row(
            row_data,
            data_cols,
            nearby_nodes_dict,
            nodes_dict,
            col_to_idx
        )

    # Stage 2: Interpolate entire zero rows
    zero_row_mask = row_sums == 0
    zero_row_indices = np.where(zero_row_mask)[0]

    for row_idx in zero_row_indices:
        row_node_id = int(node_ids_index[row_idx])
        nearby_nodes = nearby_nodes_dict.get(row_node_id, {})

        # Find nearby nodes that are non-zero rows
        non_zero_row_ids = {int(node_ids_index[i]) for i in non_zero_row_indices}
        nearby_non_zero_rows = {
            nz_id: nearby_nodes[nz_id]
            for nz_id in non_zero_row_ids
            if nz_id in nearby_nodes
        }

        if nearby_non_zero_rows:
            coords = nodes_dict[row_node_id]

            # Vectorized distance computation
            nearby_items = list(nearby_non_zero_rows.items())
            nearby_node_ids = [item[0] for item in nearby_items]
            nearby_coords = np.array([item[1] for item in nearby_items])

            dists = haversine_vector(
                [coords] * len(nearby_coords),  # Origin point repeated N times
                nearby_coords,  # N destination points
                unit=Unit.METERS
            )

            # Avoid division by zero
            dists = np.maximum(dists, 1e-10)

            # Get rows for nearby nodes
            nearby_row_indices = np.array([row_to_idx[nz_id] for nz_id in nearby_node_ids])
            nearby_rows = pmf_array[nearby_row_indices, :]

            # IDW interpolation - weighted average of entire rows
            weights = 1.0 / dists
            pmf_array[row_idx, :] = np.average(nearby_rows, axis=0, weights=weights)

    # Convert back to Polars DataFrame
    result_df = pl.DataFrame(pmf_array, schema=data_cols)
    result_df = result_df.insert_column(0, pl.Series('node_id', node_ids_index))

    # Stage 3: Interpolate external region (node 10000) column
    if has_exclude_col:
        external_col_series = (
            exclude_col_original
            .filter(
                pl.col('node_id') != exclude_node
            )
            .select(exclude_node_str)
            .to_series()
        )

        # Convert to numpy for interpolation
        external_col_array = external_col_series.to_numpy().astype(np.float64)

        # Treat column as a "row" where destinations are the origins
        # Create mapping: node_id -> index in external_col_array
        col_node_ids = node_ids_index  # These are the "column names" for this transposed row
        external_col_to_idx = {int(node_id): idx for idx, node_id in enumerate(col_node_ids)}

        # Use the general interpolation function
        external_col_array = interpolate_row(
            external_col_array,
            [str(nid) for nid in col_node_ids],  # Convert to strings like data_cols
            nearby_nodes_dict,
            nodes_dict,
            external_col_to_idx
        )

        # Add interpolated column back
        col_position = original_data_cols.index(exclude_node_str) + 1
        result_df = result_df.insert_column(
            col_position,
            pl.Series(exclude_node_str, external_col_array)
        )

    # Stage 4: Interpolate external region (node 10000) row
    if has_exclude_row:
        row_cols = result_df.select(pl.exclude('node_id')).columns
        external_row_array = exclude_row_original.select(row_cols).to_numpy()[0].astype(np.float64)

        # Create col_to_idx mapping for external row
        external_col_to_idx = {int(col): idx for idx, col in enumerate(row_cols)}

        # Use the general interpolation function
        external_row_array = interpolate_row(
            external_row_array,
            row_cols,
            nearby_nodes_dict,
            nodes_dict,
            external_col_to_idx
        )

        # Create the external row as a DataFrame
        external_row_df = pl.DataFrame([external_row_array], schema=row_cols, orient="row")
        external_row_df = external_row_df.insert_column(0, pl.Series('node_id', [exclude_node]))
        result_df = result_df.vstack(external_row_df).sort('node_id')

    return result_df


def run(config: PreprocessingConfig, source: Source) -> None:
    """Run the interpolate data step."""
    print("Initializing the graph")
    graph = load_graph(config.graph_path)

    nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes_dict = {node_id: (row["y"], row["x"]) for node_id, row in sorted(nodes.iterrows(), key=lambda item: item[0])}

    print(f"Building nearby nodes dictionary with radius {config.interpolation_radius}m")
    nearby_nodes_dict = {
        node_id: nodes_within_radius(node_id, nodes_dict, config.user_radius)
        for node_id in tqdm(nodes_dict, desc="Building Nearby Nodes", dynamic_ncols=True)
    }

    print("\nBuilding the PMF matrices")
    tbar = tqdm(
        total=len(config.days_of_week) * config.num_time_slots,
        desc="Interpolating Rate Matrices",
        dynamic_ncols=True,
    )

    for day in config.days_of_week:
        for timeslot in range(config.num_time_slots):
            # Update pbar description with current weekday and timeslot
            tbar.set_description(f"Interpolating {day} timeslot {timeslot}")

            matrix_path = os.path.join(
                config.data_path,
                "matrices",
                source.month_str,
                day.lower()
            )
            rate_matrix_file = os.path.join(matrix_path, f"{str(timeslot).zfill(2)}-rate-matrix.csv")

            if not os.path.exists(rate_matrix_file):
                print(f"Rate matrix not found: {rate_matrix_file}, skipping")
                tbar.update(1)
                continue

            # Read CSV with Polars
            rate_matrix = pl.read_csv(rate_matrix_file)

            rate_matrix = rate_matrix.with_columns(pl.col('node_id').cast(pl.Int64))

            rate_matrix = reorder_df(rate_matrix, 'node_id')

            # Build PMF matrix (handles all 4 stages internally)
            pmf_matrix = build_pmf_matrix(rate_matrix, nearby_nodes_dict, nodes_dict)

            # Normalize to PMF
            data_array = pmf_matrix.select(pl.exclude('node_id')).to_numpy()
            total_sum = math.fsum(data_array.flatten())

            if total_sum > 0:
                data_cols = pmf_matrix.select(pl.exclude('node_id')).columns
                pmf_matrix = pmf_matrix.with_columns([
                    (pl.col(col) / total_sum).alias(col) for col in data_cols
                ])

            # Save PMF matrix
            pmf_matrix.write_csv(os.path.join(matrix_path, f"{str(timeslot).zfill(2)}-pmf-matrix.csv"))

            tbar.update(1)

    tbar.close()

    print("\nInterpolation complete!")


def main():
    """CLI entry point for interpolate_data."""
    parser = argparse.ArgumentParser(description="Interpolate rate matrices to build PMF matrices.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--source", type=str, required=True, help="Source id from sources.json.")
    parser.add_argument("--sources-json", type=str, default="core/sources.json")
    parser.add_argument("--radius", type=int, default=500, help="Interpolation radius in meters.")

    args = parser.parse_args()

    src = get_source(args.sources_json, args.source)
    config = PreprocessingConfig(
        source_id=src.id,
        sources_json=args.sources_json,
        data_path=args.data_path,
        interpolation_radius=args.radius,
    )
    run(config, src)


if __name__ == "__main__":
    main()
