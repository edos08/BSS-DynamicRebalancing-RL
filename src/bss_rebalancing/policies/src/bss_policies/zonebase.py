"""Shared machinery of the zone-based baselines (issues #152, #153).

Zones (:mod:`bss_sim.zones`; ``"auto"``: a 300 m grid in dockless mode, station catchments in
docked mode), target bikes per zone from the forecast, where to stop, and the service commands:

* pick-ups go where the zone's bikes are (its node with the most parked bikes) and take the nearest;
* drops go to the zone's busiest node and are spread over its ``spread`` highest-demand nodes within
  ``radius_m``: putting all bikes on one node serves fewer users, who only accept short walks;
* optional battery swaps of e-bikes below ``swap_below`` of their range in the zone.
"""

from __future__ import annotations

import numpy as np

from bss_policies.base import Controller
from bss_policies.planning import Visit, zone_targets
from bss_sim.commands import Drop, GoTo, PickUp, SwapBattery
from bss_sim.state import ELECTRIC, PARKED
from bss_sim.zones import grid_zones, station_zones


class ZoneController(Controller):
    def __init__(self, zones: str = "auto", cell_m: float = 300.0, min_per_zone: int = 1, radius_m: float = 250.0,
                 swap_below: float | None = None, spread: int = 5, horizon_h: float = 3.0):
        if zones not in ("auto", "grid", "stations"):
            raise ValueError("zones must be 'auto', 'grid' or 'stations'")
        self.zones_kind = zones
        self.cell_m = cell_m
        self.min_per_zone = min_per_zone
        self.radius_m = radius_m
        self.swap_below = swap_below
        self.spread = spread
        self.horizon_s = horizon_h * 3600.0
        self.zones = None

    def describe(self) -> dict:
        return {"name": self.name, "zones": self.zones.name if self.zones else self.zones_kind,
                "min_per_zone": self.min_per_zone, "radius_m": self.radius_m, "swap_below": self.swap_below,
                "spread": self.spread, "horizon_h": self.horizon_s / 3600}

    def setup_zones(self, sim) -> None:
        if self.zones is None or self.zones.node_to_zone.shape[0] != sim.scenario.n_nodes:
            kind = self.zones_kind
            if kind == "auto":
                kind = "stations" if sim.config.mode == "docked" else "grid"
            self.zones = station_zones(sim.scenario) if kind == "stations" else grid_zones(sim.scenario, self.cell_m)
            self._weights = sim.scenario.departure_weights()
        try:
            sim.add_partition(self.zones.name, self.zones.node_to_zone)
        except ValueError:      # registered in an earlier episode on the same simulator
            pass
        self._docks = None
        if sim.config.mode == "docked":
            cap = sim.scenario.node_capacity()
            z = self.zones.node_to_zone
            self._docks = np.bincount(z[z >= 0], weights=cap[z >= 0], minlength=self.zones.n).astype(np.int64)

    def surplus(self, sim, obs, horizon_s: float | None = None) -> np.ndarray:
        """Bikes per zone minus target (> 0: bikes to take away, < 0: bikes to bring)."""
        z = self.zones
        bikes = obs.partitions[z.name].bikes
        fc = sim.forecast(horizon_s or self.horizon_s, z.name)
        total = int(bikes.sum() + sum(t.load for t in obs.trucks))
        target = zone_targets(fc.departures, fc.arrivals, total, self.min_per_zone, self._docks)
        return bikes - target

    def stops(self, obs, surplus: np.ndarray) -> np.ndarray:
        """Pick-ups at the zone's fullest node, drops at its busiest node."""
        z = self.zones
        nodes = np.flatnonzero(z.node_to_zone >= 0)
        order = np.lexsort((-obs.bikes[nodes], z.node_to_zone[nodes]))
        zz = z.node_to_zone[nodes][order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = zz[1:] != zz[:-1]
        fullest = z.stop.astype(np.int64).copy()
        fullest[zz[first]] = nodes[order][first]
        return np.where(surplus > 0, fullest, z.stop).astype(np.int64)

    def commands(self, sim, obs, v: Visit, node: int, stock: int) -> list:
        cmds = [GoTo(int(node))]
        if v.qty > 0:
            cmds.append(PickUp(v.qty, policy="nearest", radius_m=self.radius_m))
        else:
            cmds.append(Drop(-v.qty, at=self._drop_nodes(sim, v.zone, int(node))))
        n_swap = self._to_swap(sim, obs, v.zone)
        if n_swap and stock > 0:
            cmds.append(SwapBattery(n_swap, radius_m=self.radius_m))
        return cmds

    def _drop_nodes(self, sim, zone: int, stop: int) -> tuple[int, ...] | None:
        if self.spread <= 1:
            return None
        near = [v for v in sim.nodes_within(stop, self.radius_m) if self.zones.node_to_zone[v] == zone]
        near.sort(key=lambda v: -self._weights[v])
        chosen = tuple(int(v) for v in near[:self.spread] if self._weights[v] > 0)
        return chosen or None

    def _to_swap(self, sim, obs, zone: int) -> int:
        if self.swap_below is None:
            return 0
        limit = self.swap_below * sim.config.fleet.battery_range_km
        low = (obs.bike_status == PARKED) & (obs.bike_kind == ELECTRIC) & (obs.bike_battery_km < limit)
        idx = np.flatnonzero(low)
        return int((self.zones.node_to_zone[obs.bike_node[idx]] == zone).sum())
