"""Scenario bundle builder for the BSS simulator v2 (see docs/sim-v2/SPEC.md §4).

This package turns raw data (BlueBikes trips, OSM networks, traffic samples) into a versioned
scenario bundle that the simulator reads. It never imports the simulator or the legacy packages.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
