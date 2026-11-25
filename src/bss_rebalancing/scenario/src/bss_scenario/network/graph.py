"""Load OSM street networks (GraphML written by OSMnx) into plain node and edge tables (issue #69).

No OSMnx at run time: the GraphML is read with NetworkX and turned into Polars tables with a
dense, deterministic node index (sorted by OSM id). Only the **largest strongly connected
component** is kept, so every shortest path in the bundle is finite; the dropped nodes are
reported.

Coordinates are also given in metres (``x`` east, ``y`` north) from a local equirectangular
projection centred on the network. Over a city (< 10 km) its distortion is far below the
accuracy of the data (< 0.1 %), and it avoids a projection library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

EARTH_RADIUS_M = 6_371_008.8

#: Rank used to pick one class when OSM gives several (lower = more important road).
HIGHWAY_RANK = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
    "motorway_link": 5, "trunk_link": 5, "primary_link": 6, "secondary_link": 7, "tertiary_link": 8,
    "unclassified": 9, "residential": 10, "living_street": 11, "busway": 12, "service": 13,
    "cycleway": 14, "path": 15, "track": 16, "pedestrian": 17, "footway": 18, "steps": 19,
}

NODE_SCHEMA = {"node_idx": pl.Int32(), "osm_id": pl.Int64(), "lat": pl.Float64(), "lon": pl.Float64(),
               "x": pl.Float64(), "y": pl.Float64()}
EDGE_SCHEMA = {"u": pl.Int32(), "v": pl.Int32(), "length_m": pl.Float64(), "highway": pl.String(),
               "maxspeed": pl.String(), "oneway": pl.Boolean(), "name": pl.String()}


@dataclass(frozen=True)
class Projection:
    """Local equirectangular projection around (lat0, lon0)."""

    lat0: float
    lon0: float

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = np.radians(lon - self.lon0) * EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return x, y


@dataclass(frozen=True)
class Network:
    nodes: pl.DataFrame  # NODE_SCHEMA, node_idx == row number
    edges: pl.DataFrame  # EDGE_SCHEMA, one row per directed (u, v)
    projection: Projection
    dropped_nodes: int

    @property
    def n(self) -> int:
        return self.nodes.height


def _first_value(value) -> str | None:
    """OSMnx stores multi-valued tags as "['a', 'b']"; return the list of values."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        parts = [p.strip(" '\"") for p in text[1:-1].split(",")]
        return ";".join(p for p in parts if p) or None
    return text or None


def main_highway(value) -> str:
    """The most important road class among the values of a ``highway`` tag."""
    values = (_first_value(value) or "unclassified").split(";")
    return min(values, key=lambda h: (HIGHWAY_RANK.get(h, 99), h))


def _bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def load_graphml(path: str | Path, projection: Projection | None = None) -> Network:
    """Read a GraphML street network and return its largest strongly connected component.

    Pass the same ``projection`` to every network of a bundle so their ``x``/``y`` agree; by default
    it is centred on this network.
    """
    graph = nx.read_graphml(path)
    if not graph.is_directed():
        raise ValueError(f"{path}: expected a directed graph")
    largest = max(nx.strongly_connected_components(graph), key=len)
    dropped = graph.number_of_nodes() - len(largest)

    osm_ids = sorted(int(n) for n in largest)
    index = {str(o): i for i, o in enumerate(osm_ids)}
    lat = np.array([float(graph.nodes[str(o)]["y"]) for o in osm_ids])
    lon = np.array([float(graph.nodes[str(o)]["x"]) for o in osm_ids])
    if projection is None:
        projection = Projection(lat0=float(np.round(lat.mean(), 6)), lon0=float(np.round(lon.mean(), 6)))
    x, y = projection.to_xy(lat, lon)
    nodes = pl.DataFrame({"node_idx": np.arange(len(osm_ids), dtype=np.int32), "osm_id": osm_ids,
                          "lat": lat, "lon": lon, "x": x, "y": y}, schema=NODE_SCHEMA)

    best: dict[tuple[int, int], tuple] = {}
    for u, v, data in graph.edges(data=True):
        if u not in index or v not in index or u == v:
            continue
        key = (index[u], index[v])
        row = (key[0], key[1], float(data["length"]), main_highway(data.get("highway")),
               _first_value(data.get("maxspeed")), _bool(data.get("oneway", False)), _first_value(data.get("name")))
        if key not in best or row[2] < best[key][2]:
            best[key] = row
    edges = pl.DataFrame([best[k] for k in sorted(best)], schema=EDGE_SCHEMA, orient="row")
    return Network(nodes=nodes, edges=edges, projection=projection, dropped_nodes=dropped)
