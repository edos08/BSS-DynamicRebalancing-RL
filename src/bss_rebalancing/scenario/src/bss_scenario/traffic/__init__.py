"""Truck travel-time data: OD sampling, routing-service collection, congestion fitting (spec §9)."""

from bss_scenario.traffic.sample import ODSampleConfig, build_od_sample, endpoint_nodes, sample_od_pairs
from bss_scenario.traffic.tomtom import CollectionPlan, collect

__all__ = ["CollectionPlan", "ODSampleConfig", "build_od_sample", "collect", "endpoint_nodes", "sample_od_pairs"]
