"""Build the ``network/`` part of a scenario bundle (issues #14, #70, #71; spec §4).

Outputs (``out_dir``):

=========================  ====================================================================
``bike_nodes.parquet``     node_idx, osm_id, lat, lon, x, y (metres, shared projection)
``bike_edges.parquet``     u, v, length_m, highway, maxspeed, oneway, name
``bike_dist.npy``          float32 [N, N] directed shortest paths on the bike network (m)
``bike_walk_nbrs.npz``     CSR walking neighbours (undirected) up to ``walk_radius_max_m``
``drive_nodes.parquet``    as bike nodes, drive network without busways
``drive_edges.parquet``    as bike edges + speed_kmh, speed_source, freeflow_s
``drive_time.npy``         float32 [M, M] free-flow driving time (s), one-way respected
``drive_dist.npy``         float32 [M, M] length (m) of those fastest paths
``drive_time_by_class.npy`` float32 [C, M, M] free-flow time of those paths on each road class
                           (``ROAD_CLASSES``: major, secondary, local); sums to ``drive_time``
``drive_nodes.npz``        node_idx order: osm_id, lat, lon, x, y (for the simulator, no Parquet)
``bike_to_drive.npz``      node: int32 [N] nearest drive node of each bike node; dist_m: float32 [N]
``network_report.json``    sizes, dropped nodes, speed coverage, projection, build times
=========================  ====================================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bss_scenario.network.graph import Network, load_graphml
from bss_scenario.network.paths import all_pairs, neighbour_stats, path_sums, walk_neighbours
from bss_scenario.traffic.freeflow import EXCLUDED_FOR_VANS, ROAD_CLASSES, edge_speeds, road_class, speed_report


def nearest_nodes(src: Network, dst: Network) -> tuple[np.ndarray, np.ndarray]:
    """For every node of ``src``, the nearest node of ``dst`` (straight line, metres) and its distance.

    Both networks must share a projection.
    """
    if src.projection != dst.projection:
        raise ValueError("networks use different projections")
    tree = cKDTree(dst.nodes.select("x", "y").to_numpy())
    dist, idx = tree.query(src.nodes.select("x", "y").to_numpy(), k=1)
    return idx.astype(np.int32), dist.astype(np.float32)


def build_networks(bike_graphml: str | Path, drive_graphml: str | Path, out_dir: str | Path,
                   walk_radius_max_m: float = 800.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.perf_counter()
    bike = load_graphml(bike_graphml)
    drive = load_graphml(drive_graphml, projection=bike.projection, exclude_highways=EXCLUDED_FOR_VANS)
    drive_edges = edge_speeds(drive.edges)
    timings["load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    bike_dist = all_pairs(bike, "length_m")
    timings["bike_dist_s"] = time.perf_counter() - t

    t = time.perf_counter()
    nbrs = walk_neighbours(bike, walk_radius_max_m)
    timings["walk_nbrs_s"] = time.perf_counter() - t

    t = time.perf_counter()
    ff = drive_edges["freeflow_s"].to_numpy()
    cls = np.array([road_class(h) for h in drive_edges["highway"].to_list()])
    extras = {"length": drive_edges["length_m"].to_numpy()}
    extras |= {c: np.where(cls == i, ff, 0.0) for i, c in enumerate(ROAD_CLASSES)}
    drive_time, sums = path_sums(drive, ff, extras)
    drive_dist = sums["length"]
    by_class = np.stack([sums[c] for c in ROAD_CLASSES])
    timings["drive_time_s"] = time.perf_counter() - t

    b2d_node, b2d_dist = nearest_nodes(bike, drive)

    bike.nodes.write_parquet(out / "bike_nodes.parquet")
    bike.edges.write_parquet(out / "bike_edges.parquet")
    np.save(out / "bike_dist.npy", bike_dist)
    nbrs.save(out / "bike_walk_nbrs.npz")
    drive.nodes.write_parquet(out / "drive_nodes.parquet")
    drive_edges.write_parquet(out / "drive_edges.parquet")
    np.save(out / "drive_time.npy", drive_time)
    np.save(out / "drive_dist.npy", drive_dist)
    np.save(out / "drive_time_by_class.npy", by_class)
    dn = drive.nodes
    np.savez(out / "drive_nodes.npz", osm_id=dn["osm_id"].to_numpy(), lat=dn["lat"].to_numpy(),
             lon=dn["lon"].to_numpy(), x=dn["x"].to_numpy(), y=dn["y"].to_numpy(),
             road_classes=np.array(ROAD_CLASSES))
    np.savez(out / "bike_to_drive.npz", node=b2d_node, dist_m=b2d_dist)

    report = {
        "projection": asdict(bike.projection),
        "bike": {"nodes": bike.n, "edges": bike.edges.height, "dropped_nodes": bike.dropped_nodes,
                 "dist_m": {"median": float(np.median(bike_dist)), "max": float(bike_dist.max())},
                 "walk_radius_max_m": walk_radius_max_m,
                 "walk_neighbours": neighbour_stats(nbrs).to_dicts()},
        "drive": {"nodes": drive.n, "edges": drive.edges.height, "dropped_nodes": drive.dropped_nodes,
                  "excluded_highways": sorted(EXCLUDED_FOR_VANS),
                  "speeds": {k: v for k, v in speed_report(drive_edges).items() if k != "default_by_class"},
                  "time_s": {"median": float(np.median(drive_time)), "max": float(drive_time.max())},
                  "dist_m": {"median": float(np.median(drive_dist)), "max": float(drive_dist.max())},
                  "time_share_by_class": {c: float(by_class[i].sum() / max(drive_time.sum(), 1e-9))
                                          for i, c in enumerate(ROAD_CLASSES)}},
        "bike_to_drive_m": {"median": float(np.median(b2d_dist)), "p99": float(np.percentile(b2d_dist, 99)),
                            "max": float(b2d_dist.max())},
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "files_mb": {},
    }
    report["files_mb"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out.iterdir())
                          if p.name != "network_report.json"}
    (out / "network_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
