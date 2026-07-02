"""
Command-line interface for the preprocessing pipeline.
"""

import argparse
import os
import pickle
import sys
import time

from preprocessing.config import PreprocessingConfig, DEFAULT_SOURCES_JSON
from preprocessing.core.utils import format_time
from preprocessing.core.sources import get_source


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="BSS Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full preprocessing pipeline
  bss-preprocess --data-path data/

  # Run specific steps only
  bss-preprocess --data-path data/ --steps download,preprocess,interpolate

  # Skip certain steps
  bss-preprocess --data-path data/ --skip download

  # Plot the graph only (no preprocessing)
  bss-preprocess --data-path data/ --plot graph

  # Plot graph with cell grid
  bss-preprocess --data-path data/ --plot grid

  # Plot graph with cell grid and cell numbers
  bss-preprocess --data-path data/ --plot grid-numbered
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to the data directory (default: data/)",
    )

    parser.add_argument(
        "--source",
        type=str, required=True,
        help="Source id from sources.json (e.g. 'bluebikes', 'citibike')"
    )

    parser.add_argument(
        "--sources-json",
        type=str,
        default=DEFAULT_SOURCES_JSON,
        help="Path to sources.json (default: bundled with the package)"
    )

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to run. Options: download, preprocess, interpolate, grid, distance, nodes, ev_matrices",
    )

    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated list of steps to skip",
    )

    parser.add_argument(
        "--cell-size",
        type=int,
        default=300,
        help="Cell size in meters for truck grid (default: 300)",
    )

    parser.add_argument(
        "--plot",
        type=str,
        choices=["graph", "grid", "grid-numbered"],
        default=None,
        help="Plot mode (skips preprocessing): 'graph' = base graph only, 'grid' = graph + cells, 'grid-numbered' = graph + cells + cell IDs",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def run_plot_mode(config: PreprocessingConfig, plot_mode: str) -> None:
    """
    Run in plot-only mode without preprocessing.

    Parameters:
        config: The preprocessing configuration.
        plot_mode: One of 'graph', 'grid', or 'grid-numbered'.
    """
    from preprocessing.core.graph import load_graph
    from preprocessing.core.plotting import plot_graph, plot_graph_with_grid

    # Create plots output directory
    plots_path = os.path.join(config.data_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    print(f"Loading graph from {config.graph_path}...")
    try:
        graph = load_graph(config.graph_path)
    except FileNotFoundError:
        print(f"Error: Graph file not found at {config.graph_path}")
        print("Please run the preprocessing pipeline first, or check the data path.")
        sys.exit(1)

    if plot_mode == "graph":
        print("Plotting base graph...")
        output_file = os.path.join(plots_path, "graph.png")
        plot_graph(graph, save_path=output_file)
        print(f"\nPlot saved to: {output_file}")

    elif plot_mode in ("grid", "grid-numbered"):
        # Load cell data
        cell_data_path = os.path.join(config.data_path, config.cell_data_path)
        if not os.path.exists(cell_data_path):
            print(f"Error: Cell data not found at {cell_data_path}")
            print("Please run the 'grid' preprocessing step first.")
            sys.exit(1)

        print(f"Loading cell data from {cell_data_path}...")
        with open(cell_data_path, "rb") as f:
            cell_dict = pickle.load(f)

        plot_number_cells = (plot_mode == "grid-numbered")
        print(f"Plotting graph with grid (numbered={plot_number_cells})...")

        if plot_number_cells:
            output_file = os.path.join(plots_path, "grid_numbered.png")
        else:
            output_file = os.path.join(plots_path, "grid.png")
        plot_graph_with_grid(
            graph,
            cell_dict,
            plot_center_nodes=True,
            plot_number_cells=plot_number_cells,
            save_path=output_file,
        )
        print(f"\nPlot saved to: {output_file}")


def run_pipeline(config: PreprocessingConfig, source, steps: list, verbose: bool = False):
    """Run the preprocessing pipeline with the given configuration."""
    from datetime import datetime, timezone
    from preprocessing.steps import (
        create_ev_matrices,
        download_trips,
        preprocess_data,
        interpolate_data,
        preprocess_truck_grid,
        preprocess_distance_matrix,
        preprocess_nodes_dictionary,
    )
    from preprocessing.core.manifest import write_manifest

    step_mapping = {
        "download": ("Downloading trip data", download_trips.run),
        "preprocess": ("Preprocessing trip data", preprocess_data.run),
        "interpolate": ("Interpolating data", interpolate_data.run),
        "grid": ("Preprocessing truck grid", preprocess_truck_grid.run),
        "distance": ("Preprocessing distance matrix", preprocess_distance_matrix.run),
        "nodes": ("Preprocessing nodes dictionary", preprocess_nodes_dictionary.run),
        "ev_matrices": ("Creating EV matrices", create_ev_matrices.run),
    }

    # Steps that have warnings
    step_warnings = {
        "preprocess": "⚠️  WARNING: This operation can take a long time depending on the amount of data.",
    }

    # Ensure utils directory exists
    os.makedirs(config.utils_path, exist_ok=True)

    # Start timing
    pipeline_start_dt = datetime.now(timezone.utc)
    pipeline_start = time.time()
    step_times = {}
    completed_steps = []

    for step_name in steps:
        if step_name not in step_mapping:
            print(f"Warning: Unknown step '{step_name}', skipping...")
            continue

        description, step_func = step_mapping[step_name]
        print(f"\n{'=' * 60}")
        print(f"Step: {description}")
        print(f"{'=' * 60}")

        # Show warning if applicable
        if step_name in step_warnings:
            print(step_warnings[step_name])

        # Time each step
        step_start = time.time()

        try:
            step_func(config, source)
        except Exception as e:
            print(f"Error in step '{step_name}': {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            # No manifest on failure — a manifest describing a broken run
            # is strictly worse than no manifest, it just lies with confidence.
            sys.exit(1)

        step_elapsed = time.time() - step_start
        step_times[step_name] = step_elapsed
        completed_steps.append(step_name)
        print(f"✓ Step completed in {format_time(step_elapsed)}")

    total_elapsed = time.time() - pipeline_start

    print(f"\n{'=' * 60}")
    print("Preprocessing pipeline completed successfully!")
    print(f"{'=' * 60}")

    # Display timing summary
    print("\n⏱️  Timing Summary:")
    print(f"{'=' * 60}")
    for step_name, elapsed in step_times.items():
        percentage = (elapsed / total_elapsed) * 100
        print(f"  {step_name:15} : {format_time(elapsed):>12}  ({percentage:5.1f}%)")
    print(f"{'=' * 60}")
    print(f"  {'Total':15} : {format_time(total_elapsed):>12}")
    print(f"{'=' * 60}")

    # Only write the manifest once every requested step actually finished.
    write_manifest(config, source, completed_steps, pipeline_start_dt)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    src = get_source(args.sources_json, args.source)
    config = PreprocessingConfig(
        source_id=src.id,
        sources_json=args.sources_json,
        data_path=args.data_path,
        cell_size=src.cell_size,
    )

    # Handle plot mode (skips preprocessing)
    if args.plot:
        print(f"BSS Preprocessing - Plot Mode")
        print(f"Data path: {config.data_path}")
        print(f"Plot type: {args.plot}")
        run_plot_mode(config, args.plot)
        return

    # Determine which steps to run
    all_steps = ["download", "preprocess", "interpolate", "grid", "distance", "nodes", "ev_matrices"]

    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    else:
        steps = all_steps

    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(",")]
        steps = [s for s in steps if s not in skip_steps]

    print(f"BSS Preprocessing Pipeline")
    print(f"Source: {src.id}")
    print(f"Data path: {config.data_path}")
    print(f"Years: {src.years}, Months: {src.months}")
    print(f"Steps to run: {', '.join(steps)}")

    run_pipeline(config, src, steps, verbose=args.verbose)


if __name__ == "__main__":
    main()