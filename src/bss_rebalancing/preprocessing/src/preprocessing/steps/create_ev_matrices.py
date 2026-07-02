"""
Create static EV consumption and velocity matrices.

These matrices contain lookup values for electric vehicle energy consumption (kWh/km)
and velocity (km/h) based on hour of day and day of week. Values reflect typical
urban traffic patterns.
"""

import argparse
import os

import pandas as pd

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.sources import Source, get_source


# EV consumption matrix (kWh/km) - varies by traffic conditions
EV_CONSUMPTION_DATA = {
    "hour": list(range(24)),
    "sunday": [0.22, 0.22, 0.22, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.22,
               0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.21, 0.21],
    "monday": [0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.22, 0.23, 0.22, 0.22, 0.22,
               0.22, 0.22, 0.23, 0.23, 0.23, 0.24, 0.23, 0.22, 0.22, 0.22, 0.21, 0.21],
    "tuesday": [0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.23, 0.24, 0.23, 0.22, 0.22,
                0.22, 0.23, 0.23, 0.24, 0.25, 0.26, 0.24, 0.22, 0.22, 0.22, 0.21, 0.21],
    "wednesday": [0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.23, 0.24, 0.23, 0.23, 0.22,
                  0.23, 0.23, 0.24, 0.24, 0.25, 0.26, 0.24, 0.23, 0.22, 0.22, 0.21, 0.21],
    "thursday": [0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.23, 0.24, 0.23, 0.22, 0.22,
                 0.23, 0.23, 0.24, 0.25, 0.25, 0.26, 0.24, 0.23, 0.22, 0.22, 0.21, 0.21],
    "friday": [0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22,
               0.23, 0.23, 0.24, 0.24, 0.24, 0.24, 0.23, 0.23, 0.22, 0.22, 0.22, 0.22],
    "saturday": [0.21, 0.21, 0.22, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.22,
                 0.22, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.22, 0.22, 0.22, 0.22, 0.22],
}

# EV velocity matrix (km/h) - varies by traffic conditions
EV_VELOCITY_DATA = {
    "hour": list(range(24)),
    "sunday": [39, 40, 41, 47, 53, 54, 50, 50, 48, 45, 40, 36,
               33, 32, 31, 31, 31, 32, 33, 36, 38, 39, 40, 41],
    "monday": [44, 44, 45, 52, 56, 50, 39, 31, 29, 32, 35, 35,
               35, 34, 30, 28, 27, 27, 31, 36, 37, 39, 39, 40],
    "tuesday": [42, 44, 45, 51, 56, 49, 37, 29, 26, 28, 30, 32,
                33, 32, 28, 25, 23, 21, 25, 32, 36, 37, 37, 38],
    "wednesday": [40, 42, 43, 50, 55, 49, 37, 29, 26, 27, 30, 32,
                  32, 31, 26, 24, 23, 21, 25, 31, 35, 36, 36, 37],
    "thursday": [40, 42, 44, 50, 55, 49, 37, 29, 26, 28, 30, 32,
                 32, 30, 26, 23, 22, 21, 24, 30, 34, 35, 35, 36],
    "friday": [39, 41, 42, 49, 54, 49, 38, 32, 31, 33, 33, 32,
               31, 29, 25, 24, 24, 25, 28, 31, 34, 35, 34, 35],
    "saturday": [37, 39, 40, 47, 53, 54, 50, 48, 45, 41, 37, 34,
                 31, 30, 29, 29, 29, 30, 30, 32, 35, 37, 37, 38],
}


def create_ev_consumption_matrix() -> pd.DataFrame:
    """
    Create the EV consumption matrix.

    Returns:
        DataFrame with EV consumption values (kWh/km) by hour and day of week.
    """
    return pd.DataFrame(EV_CONSUMPTION_DATA)


def create_ev_velocity_matrix() -> pd.DataFrame:
    """
    Create the EV velocity matrix.

    Returns:
        DataFrame with EV velocity values (km/h) by hour and day of week.
    """
    return pd.DataFrame(EV_VELOCITY_DATA)


def run(config: PreprocessingConfig, source: Source) -> None:
    """
    Run the EV matrices creation step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    # Create and save consumption matrix
    consumption_matrix = create_ev_consumption_matrix()
    consumption_path = os.path.join(config.utils_path, "ev_consumption_matrix.csv")
    consumption_matrix.to_csv(consumption_path, index=False)
    print(f"Saved EV consumption matrix to {consumption_path}")

    # Create and save velocity matrix
    velocity_matrix = create_ev_velocity_matrix()
    velocity_path = os.path.join(config.utils_path, "ev_velocity_matrix.csv")
    velocity_matrix.to_csv(velocity_path, index=False)
    print(f"Saved EV velocity matrix to {velocity_path}")

    print(f"EV consumption matrix shape: {consumption_matrix.shape}")
    print(f"EV velocity matrix shape: {velocity_matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Create static EV consumption and velocity matrices.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--source", type=str, required=True, help="Source id from sources.json (unused, kept for CLI symmetry).")
    parser.add_argument("--sources-json", type=str, default="core/sources.json")

    args = parser.parse_args()

    src = get_source(args.sources_json, args.source)
    config = PreprocessingConfig(source_id=src.id, sources_json=args.sources_json, data_path=args.data_path)
    run(config, src)


if __name__ == "__main__":
    main()