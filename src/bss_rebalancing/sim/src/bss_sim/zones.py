"""Zones: partitions of the bike nodes for controllers (issue #149, spec §5.4, D4).

A :class:`Zones` is a partition (``node_to_zone``, -1 = no zone) plus one **stop** per zone: the
bike node where a truck serves the zone (it drives to the drive node nearest to it). Register it
with ``sim.add_partition(zones.name, zones.node_to_zone)``.

* :func:`grid_zones`: the thesis's square grid (300 m by default) over the bike network; empty
  cells are dropped, zones are numbered row by row.
* :func:`station_zones`: the catchment of each station (each node goes to the station that gives it
  the largest share of its spread demand); the stop is the station's node.

Stops: ``"demand"`` (default) is the node with the most expected departures in the zone (falling back
to ``"center"`` when the zone has no demand), ``"center"`` the node closest to the zone's centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Zones:
    name: str
    node_to_zone: np.ndarray     # int64 [nodes], -1 = no zone
    stop: np.ndarray             # int32 [Z] bike node served by trucks
    center: np.ndarray           # float64 [Z, 2] centroid (x, y) of the zone's nodes

    @property
    def n(self) -> int:
        return len(self.stop)

    def nodes(self, z: int) -> np.ndarray:
        return np.flatnonzero(self.node_to_zone == z)


def _stops(scenario, node_to_zone: np.ndarray, n: int, how: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = scenario.bike_x, scenario.bike_y
    keep = node_to_zone >= 0
    z = node_to_zone[keep]
    count = np.bincount(z, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):   # a zone can be empty (two stations on one node)
        cx = np.bincount(z, weights=x[keep], minlength=n) / count
        cy = np.bincount(z, weights=y[keep], minlength=n) / count
    d2 = (x - cx[np.maximum(node_to_zone, 0)]) ** 2 + (y - cy[np.maximum(node_to_zone, 0)]) ** 2
    weight = scenario.departure_weights() if how == "demand" else np.zeros(len(x))
    stop = np.empty(n, dtype=np.int32)
    idx = np.flatnonzero(keep)
    # per zone: max demand weight, ties (and zones without demand) by distance to the centroid
    order = np.lexsort((d2[idx], -weight[idx], z))
    first = np.ones(len(order), dtype=bool)
    first[1:] = z[order][1:] != z[order][:-1]
    stop[z[order][first]] = idx[order][first]
    return stop, np.column_stack([cx, cy])


def grid_zones(scenario, cell_m: float = 300.0, stop: str = "demand", name: str | None = None) -> Zones:
    if cell_m <= 0:
        raise ValueError("cell_m must be > 0")
    x, y = scenario.bike_x, scenario.bike_y
    ix = np.floor((x - x.min()) / cell_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_m).astype(np.int64)
    cell = iy * (ix.max() + 1) + ix
    used, node_to_zone = np.unique(cell, return_inverse=True)   # row by row, empty cells dropped
    node_to_zone = node_to_zone.astype(np.int64)
    stops, center = _stops(scenario, node_to_zone, len(used), stop)
    return Zones(name or f"grid{cell_m:g}", node_to_zone, stops, center)


def station_zones(scenario, name: str = "stations") -> Zones:
    sp = scenario.demand.spreader
    n_nodes = scenario.n_nodes
    node_to_zone = np.full(n_nodes, -1, dtype=np.int64)
    place = np.repeat(np.arange(sp.n), np.diff(sp.indptr))
    order = np.argsort(sp.probs, kind="stable")          # ascending: the largest share is written last
    node_to_zone[sp.nodes[order]] = place[order]
    node_to_zone[sp.station_nodes] = np.arange(sp.n)     # a station's own node is in its zone
    _, center = _stops(scenario, node_to_zone, sp.n, "center")
    empty = ~np.isfinite(center[:, 0])     # a station sharing its node with another: centred on it
    center[empty] = np.column_stack([scenario.bike_x, scenario.bike_y])[sp.station_nodes[empty]]
    return Zones(name, node_to_zone, sp.station_nodes.astype(np.int32).copy(), center)
