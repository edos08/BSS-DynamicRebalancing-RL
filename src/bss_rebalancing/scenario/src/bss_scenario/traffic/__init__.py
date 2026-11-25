"""Truck travel-time data: OD sampling, routing-service collection, congestion fitting (spec §9)."""

from bss_scenario.traffic.sample import ODSampleConfig, build_od_sample, endpoint_nodes, sample_od_pairs

__all__ = ["ODSampleConfig", "build_od_sample", "endpoint_nodes", "sample_od_pairs"]
