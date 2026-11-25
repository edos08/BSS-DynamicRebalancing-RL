"""Spatial spreading of station demand over bike-network nodes (issue #82, spec §6.2).

This replaces the legacy "interpolation". Demand is estimated per place (in-city station); in
**dockless** mode the simulator needs it per bike node. A spreader gives, for every in-city
place, a probability distribution over nodes:

* **Kernel** (default): each bike node joins the **catchment** of its nearest in-city station by
  walking distance, if that station is within ``radius_m``. Inside a catchment, the weight of a
  node at walking distance ``w`` from the station is ``exp(-w / scale_m)``, normalised **per
  station**, so a station's demand is redistributed, never created or lost (the legacy IDW
  replicated rates into empty OD entries and inflated demand where nodes are dense).
* **Station** (docked mode): all of a place's demand at its station's node.

Stored as CSR (``indptr`` [S + 1], ``nodes``, ``probs``) in ``demand/spread.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bss_scenario.network.paths import Neighbours


@dataclass(frozen=True)
class Spread:
    mode: str                 # "kernel" | "station"
    indptr: np.ndarray        # int64 [S + 1]
    nodes: np.ndarray         # int32 [nnz]
    probs: np.ndarray         # float32 [nnz], sum to 1 per station
    radius_m: float
    scale_m: float

    @property
    def n_stations(self) -> int:
        return len(self.indptr) - 1

    def of(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.indptr[s], self.indptr[s + 1]
        return self.nodes[a:b], self.probs[a:b]

    def save(self, path: str | Path) -> None:
        np.savez(path, mode=np.array(self.mode), indptr=self.indptr, nodes=self.nodes, probs=self.probs,
                 radius_m=np.float64(self.radius_m), scale_m=np.float64(self.scale_m))

    @classmethod
    def load(cls, path: str | Path) -> Spread:
        with np.load(path) as z:
            return cls(str(z["mode"]), z["indptr"], z["nodes"], z["probs"], float(z["radius_m"]), float(z["scale_m"]))


def station_spread(station_nodes: np.ndarray) -> Spread:
    s = len(station_nodes)
    return Spread("station", np.arange(s + 1, dtype=np.int64), station_nodes.astype(np.int32),
                  np.ones(s, dtype=np.float32), 0.0, 0.0)


def kernel_spread(station_nodes: np.ndarray, walk: Neighbours, radius_m: float = 400.0,
                  scale_m: float = 150.0) -> Spread:
    """Nearest-station catchments within ``radius_m`` with exponential weights, normalised per station."""
    if radius_m > walk.max_radius_m:
        raise ValueError(f"radius {radius_m} m exceeds the stored walking radius {walk.max_radius_m} m")
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    st_idx, node_idx, dist = [], [], []
    for s, node in enumerate(station_nodes):
        nbrs, d = walk.of(int(node))
        keep = d <= radius_m
        st_idx.append(np.full(keep.sum(), s))
        node_idx.append(nbrs[keep])
        dist.append(d[keep])
    st_idx, node_idx, dist = np.concatenate(st_idx), np.concatenate(node_idx), np.concatenate(dist)

    # nearest station per node (ties -> lower station index)
    order = np.lexsort((st_idx, dist, node_idx))
    st_idx, node_idx, dist = st_idx[order], node_idx[order], dist[order]
    first = np.ones(len(node_idx), dtype=bool)
    first[1:] = node_idx[1:] != node_idx[:-1]
    st_idx, node_idx, dist = st_idx[first], node_idx[first], dist[first]
    # a station always keeps its own node, even if another station shares it
    own = np.isin(np.arange(len(station_nodes)), st_idx[dist == 0])
    for s in np.flatnonzero(~own):
        st_idx = np.append(st_idx, s)
        node_idx = np.append(node_idx, station_nodes[s])
        dist = np.append(dist, 0.0)

    w = np.exp(-dist / scale_m)
    order = np.lexsort((node_idx, st_idx))
    st_idx, node_idx, w = st_idx[order], node_idx[order], w[order]
    counts = np.bincount(st_idx, minlength=len(station_nodes))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    totals = np.add.reduceat(w, indptr[:-1])
    probs = (w / np.repeat(totals, counts)).astype(np.float32)
    return Spread("kernel", indptr, node_idx.astype(np.int32), probs, float(radius_m), float(scale_m))


def spread_report(spread: Spread, n_nodes: int) -> dict:
    sizes = np.diff(spread.indptr)
    covered = len(np.unique(spread.nodes))
    top = np.array([spread.of(s)[1].max() for s in range(spread.n_stations)])
    return {"mode": spread.mode, "radius_m": spread.radius_m, "scale_m": spread.scale_m,
            "catchment_nodes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
            "nodes_covered": covered, "nodes_total": n_nodes, "covered_share": covered / n_nodes,
            "max_node_prob": {"median": float(np.median(top)), "min": float(top.min())}}
