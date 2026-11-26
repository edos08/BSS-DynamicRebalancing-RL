"""
Benchmark module for BSS Dynamic Rebalancing RL project.

This package provides benchmarking capabilities for evaluating static
rebalancing strategies in bike-sharing systems.
"""

from .run import main, run_benchmark, run_simulation
from .utils import (
    build_cell_graph_from_cells,
    update_cell_graph_features,
    convert_seconds_to_hours_minutes,
    set_seed
)
from .logging_config import LoggingConfig, init_logging, get_logger
from .results_manager import EpisodeResults, ResultsManager

__version__ = "1.1.0"

__all__ = [
    # Main functions
    "main",
    "run_benchmark",
    "run_simulation",
    # Utilities
    "set_seed",
    "build_cell_graph_from_cells",
    "update_cell_graph_features",
    "convert_seconds_to_hours_minutes",
    # Logging
    "LoggingConfig",
    "init_logging",
    "get_logger",
    # Results Manager
    "EpisodeResults",
    "ResultsManager",
]
