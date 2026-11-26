"""Stratified origin–destination sample for routing queries (issue #55, spec §9).

The truck travel model is fitted on travel times returned by a routing service with historical
traffic. Each OD pair costs one query per hour and day type, so the sample must be small but
cover the city: pairs are stratified by straight-line **distance band** and spread over a grid of
**areas** (origins assigned round-robin over grid cells). Endpoints are drive-network nodes on
ordinary streets (not motorway links or busways), so the routing service snaps them to a street a
van would actually use.

The sample is a pure function of the graph and the seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

EARTH_RADIUS_M = 6_371_008.8

#: Road classes whose nodes may be used as query endpoints.
ENDPOINT_HIGHWAYS = frozenset(
    {"residential", "living_street", "unclassified", "tertiary", "secondary", "primary"}
)

OD_SCHEMA: dict[str, pl.DataType] = {
    "pair_id": pl.Int32(),
    "band": pl.Int8(),
    "o_node": pl.String(),
    "o_lat": pl.Float64(),
    "o_lon": pl.Float64(),
    "o_cell": pl.Int16(),
    "d_node": pl.String(),
    "d_lat": pl.Float64(),
    "d_lon": pl.Float64(),
    "d_cell": pl.Int16(),
    "crow_m": pl.Float64(),
}


@dataclass(frozen=True)
class ODSampleConfig:
    n_pairs: int = 200
    #: Straight-line distance bands in metres, [low, high).
    bands_m: tuple[tuple[float, float], ...] = ((300, 1000), (1000, 2000), (2000, 3500), (3500, 1e9))
    grid: tuple[int, int] = (3, 3)  # rows (lat) x cols (lon)
    seed: int = 20241
    max_tries: int = 10_000

    def __post_init__(self) -> None:
        if self.n_pairs < len(self.bands_m):
            raise ValueError("n_pairs must be at least the number of bands")
        for lo, hi in self.bands_m:
            if not 0 <= lo < hi:
                raise ValueError(f"bad band {(lo, hi)}")


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _highways(value) -> set[str]:
    # osmnx stores multi-valued tags as the string "['a', 'b']"
    text = str(value).strip("[]")
    return {part.strip(" '\"") for part in text.split(",") if part.strip(" '\"")}


def endpoint_nodes(graph: nx.MultiDiGraph) -> pl.DataFrame:
    """Nodes touching at least one ordinary street, with coordinates, sorted by node id."""
    keep = set()
    for u, v, data in graph.edges(data=True):
        if _highways(data.get("highway", "")) & ENDPOINT_HIGHWAYS:
            keep.update((u, v))
    rows = [(str(n), float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])) for n in keep]
    return pl.DataFrame(rows, schema={"node": pl.String, "lat": pl.Float64, "lon": pl.Float64},
                        orient="row").sort("node")


def _cells(lat: np.ndarray, lon: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    rows, cols = grid
    r = np.minimum(((lat - lat.min()) / (np.ptp(lat) or 1) * rows).astype(int), rows - 1)
    c = np.minimum(((lon - lon.min()) / (np.ptp(lon) or 1) * cols).astype(int), cols - 1)
    return r * cols + c


def _split(n: int, k: int) -> list[int]:
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def sample_od_pairs(nodes: pl.DataFrame, config: ODSampleConfig = ODSampleConfig()) -> pl.DataFrame:
    """Draw ``config.n_pairs`` OD pairs stratified by distance band and origin area."""
    rng = np.random.default_rng(config.seed)
    lat = nodes["lat"].to_numpy()
    lon = nodes["lon"].to_numpy()
    ids = nodes["node"].to_list()
    cell = _cells(lat, lon, config.grid)
    occupied = np.unique(cell)
    by_cell = {c: np.flatnonzero(cell == c) for c in occupied}

    rows = []
    seen: set[tuple[int, int]] = set()
    for band, ((lo, hi), n_band) in enumerate(zip(config.bands_m, _split(config.n_pairs, len(config.bands_m)))):
        start = rng.integers(len(occupied))
        for k in range(n_band):
            origin_cell = occupied[(start + k) % len(occupied)]
            for _ in range(config.max_tries):
                o = rng.choice(by_cell[origin_cell])
                dist = haversine_m(lat[o], lon[o], lat, lon)
                candidates = np.flatnonzero((dist >= lo) & (dist < hi))
                if candidates.size:
                    d = int(rng.choice(candidates))
                    if (o, d) not in seen:
                        break
            else:
                raise ValueError(f"cannot find a pair in band {lo}-{hi} m from cell {origin_cell}")
            seen.add((o, d))
            rows.append((len(rows), band, ids[o], lat[o], lon[o], int(cell[o]),
                         ids[d], lat[d], lon[d], int(cell[d]), float(dist[d])))
    return pl.DataFrame(rows, schema=OD_SCHEMA, orient="row")


def build_od_sample(graphml: str | Path, out: str | Path, config: ODSampleConfig = ODSampleConfig()) -> pl.DataFrame:
    """Read a drive network (GraphML from OSMnx), sample OD pairs and write them to ``out`` (Parquet)."""
    graph = nx.read_graphml(graphml)
    sample = sample_od_pairs(endpoint_nodes(graph), config)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(out)
    return sample


def summarise(sample: pl.DataFrame, bands_m: Sequence[tuple[float, float]]) -> pl.DataFrame:
    """Pairs per band with distance range, and number of distinct origin/destination cells."""
    return (
        sample.group_by("band")
        .agg(pl.len().alias("pairs"), pl.col("crow_m").min().alias("min_m"), pl.col("crow_m").max().alias("max_m"),
             pl.col("o_cell").n_unique().alias("o_cells"), pl.col("d_cell").n_unique().alias("d_cells"))
        .sort("band")
        .with_columns(pl.col("band").map_elements(lambda b: f"{bands_m[b][0]:.0f}-{bands_m[b][1]:.0f}",
                                                  return_dtype=pl.String).alias("range_m"))
    )
