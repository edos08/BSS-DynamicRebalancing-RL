"""
Shared configuration for the preprocessing pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCES_JSON = os.path.join(_PACKAGE_DIR, "core", "sources.json")

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    source_id: str = "bluebikes"
    sources_json: str = DEFAULT_SOURCES_JSON

    # Location settings
    network_type: str = "bike"

    # Path settings
    data_path: str = "data/"
    graph_file: str = "utils/network.graphml"
    cell_data_path: str = "utils/cell_data.pkl"
    global_rates_path: str = "utils/global_rates.pkl"
    distance_matrix_path: str = "utils/distance_matrix.csv"
    nearby_nodes_path: str = "utils/nearby_nodes.pkl"

    # Nodes to remove from graph
    # nodes_to_remove: List[Tuple[float, float]] = field(default_factory=lambda: [(42.365455, -71.14254)])
    # nodes_to_remove: List[int] = field(default_factory=lambda: [330,482,54,256,36,324])
    nodes_to_remove : List[str] = field(
        default_factory=lambda: []
    )

    # Days of week to process
    days_of_week: List[str] = field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    # Grid settings
    cell_size: int = 300  # meters

    # Radius settings
    interpolation_radius: int = 500  # meters for PMF interpolation
    user_radius: int = 250  # meters for nearby nodes

    # Number of time slots per day
    num_time_slots: int = 8

    @property
    def graph_path(self) -> str:
        """Return full path to graph file."""
        return os.path.join(self.data_path, self.graph_file)

    @property
    def utils_path(self) -> str:
        """Return path to utils directory."""
        return os.path.join(self.data_path, "utils")

    @property
    def trips_path(self) -> str:
        """Return path to trips directory."""
        return os.path.join(self.data_path, "trips")


# Default configuration instance
DEFAULT_CONFIG = PreprocessingConfig()
