"""Shortest-path matrices and walking neighbourhoods (issue #70, spec §4 and §7.1).

* :func:`all_pairs` — dense ``float32 [N, N]`` directed shortest paths (metres or seconds), so the
  simulator looks distances up instead of searching graphs.
* :func:`path_sums` — the same fastest paths, plus sums of other edge attributes along them
  (e.g. length, or time on each road class), without new searches per attribute.
* :func:`walk_neighbours` — for every node, the nodes within walking reach, sorted by walking
  distance, as CSR arrays. Pedestrians ignore one-way restrictions, so walking uses the
  undirected network. Built up to ``max_radius_m``; the user model applies its own (smaller)
  radius at run time without rebuilding the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from bss_scenario.network.graph import Network

CHUNK = 512  # source rows per Dijkstra call (bounds the float64 working memory)


def adjacency(net: Network, weight: str | np.ndarray = "length_m", undirected: bool = False) -> csr_matrix:
    """Sparse adjacency with the given edge weight (column name or array aligned with ``net.edges``)."""
    w = net.edges[weight].to_numpy() if isinstance(weight, str) else np.asarray(weight, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("edge weights must be finite and positive")
    u = net.edges["u"].to_numpy()
    v = net.edges["v"].to_numpy()
    if undirected:
        u, v, w = np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([w, w])
    # duplicates (u, v) can appear when undirected: keep the minimum weight
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    first = np.ones(len(u), dtype=bool)
    first[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    return csr_matrix((w[first], (u[first], v[first])), shape=(net.n, net.n))


def all_pairs(net: Network, weight: str | np.ndarray = "length_m") -> np.ndarray:
    """Directed all-pairs shortest paths as ``float32 [N, N]``; raises if any pair is unreachable."""
    graph = adjacency(net, weight)
    out = np.empty((net.n, net.n), dtype=np.float32)
    for start in range(0, net.n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, net.n))
        out[rows] = dijkstra(graph, directed=True, indices=rows)
    if not np.isfinite(out).all():
        raise ValueError("network is not strongly connected (infinite distances)")
    return out


def path_sums(net: Network, weight: str | np.ndarray, extras: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Shortest paths by ``weight`` and, for each extra edge attribute, its sum along those paths.

    Between parallel edges the one with the smallest ``weight`` is used, as in :func:`adjacency`.
    Sums are accumulated down each source's shortest-path tree, one tree level per step
    (``acc[v] = acc[pred[v]] + x(pred[v], v)``), vectorised over a chunk of sources.
    Returns ``(float32 [N, N] weight, {name: float32 [N, N]})``.
    """
    w = net.edges[weight].to_numpy() if isinstance(weight, str) else np.asarray(weight, dtype=float)
    u, v = net.edges["u"].to_numpy(), net.edges["v"].to_numpy()
    order = np.argsort(-w, kind="stable")  # assign heaviest first: the lightest parallel edge wins
    dense = {}
    for name, x in extras.items():
        m = np.zeros((net.n, net.n))
        m[u[order], v[order]] = np.asarray(x, dtype=float)[order]
        dense[name] = m
    graph = adjacency(net, w)
    out = np.empty((net.n, net.n), dtype=np.float32)
    sums = {name: np.empty((net.n, net.n), dtype=np.float32) for name in extras}
    for start in range(0, net.n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, net.n))
        d, pred = dijkstra(graph, directed=True, indices=rows, return_predecessors=True)
        if not np.isfinite(d).all():
            raise ValueError("network is not strongly connected (infinite distances)")
        out[rows] = d
        r = np.arange(len(rows))[:, None]
        has = pred >= 0
        p = np.where(has, pred, 0)
        cols = np.broadcast_to(np.arange(net.n), pred.shape)
        for name, m in dense.items():
            step = np.where(has, m[p, cols], 0.0)
            acc = np.zeros_like(step)
            for _ in range(net.n):  # at most N tree levels; usually a few dozen
                new = np.where(has, acc[r, p] + step, 0.0)
                if np.array_equal(new, acc):
                    break
                acc = new
            sums[name][rows] = acc
    return out, sums


@dataclass(frozen=True)
class Neighbours:
    """CSR neighbour lists: node ``i``'s neighbours are ``indices[indptr[i]:indptr[i+1]]``."""

    indptr: np.ndarray   # int64 [N + 1]
    indices: np.ndarray  # int32 [nnz], sorted by distance within each row (ties by index)
    dist: np.ndarray     # float32 [nnz]
    max_radius_m: float

    def of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.dist[s:e]

    def save(self, path: str | Path) -> None:
        np.savez(path, indptr=self.indptr, indices=self.indices, dist=self.dist,
                 max_radius_m=np.float64(self.max_radius_m))

    @classmethod
    def load(cls, path: str | Path) -> Neighbours:
        with np.load(path) as z:
            return cls(z["indptr"], z["indices"], z["dist"], float(z["max_radius_m"]))


def walk_neighbours(net: Network, max_radius_m: float = 800.0) -> Neighbours:
    """Nodes within ``max_radius_m`` walking distance (undirected), the node itself first."""
    if max_radius_m <= 0:
        raise ValueError("max_radius_m must be positive")
    graph = adjacency(net, "length_m", undirected=True)
    indptr = [0]
    indices: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for start in range(0, net.n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, net.n))
        d = dijkstra(graph, directed=True, indices=rows, limit=max_radius_m)
        for r in range(len(rows)):
            near = np.flatnonzero(d[r] <= max_radius_m)
            order = np.lexsort((near, d[r, near]))
            indices.append(near[order].astype(np.int32))
            dists.append(d[r, near[order]].astype(np.float32))
            indptr.append(indptr[-1] + len(near))
    return Neighbours(np.asarray(indptr, dtype=np.int64), np.concatenate(indices), np.concatenate(dists),
                      float(max_radius_m))


def neighbour_stats(nb: Neighbours, radii=(200.0, 400.0, 800.0)) -> pl.DataFrame:
    """Median / p10 / p90 number of neighbours within each radius (for build reports)."""
    rows = []
    counts_all = np.diff(nb.indptr)
    for r in radii:
        if r > nb.max_radius_m:
            continue
        within = np.add.reduceat((nb.dist <= r).astype(np.int64), nb.indptr[:-1]) if nb.dist.size else counts_all
        within = np.where(counts_all > 0, within, 0)
        rows.append({"radius_m": r, "p10": float(np.percentile(within, 10)),
                     "median": float(np.median(within)), "p90": float(np.percentile(within, 90))})
    return pl.DataFrame(rows)
