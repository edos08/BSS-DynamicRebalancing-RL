"""Generate a tiny synthetic dataset (2x2 cells, 8 stations) with the layout the environments expect.

Cell ids include 18, the depot id hard-coded in train/validate/benchmark.
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import polars as pl
from shapely.geometry import Polygon

from gymnasium_env.simulator.cell import Cell

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def build(root: str) -> None:
    os.makedirs(os.path.join(root, "utils"), exist_ok=True)

    # 2x2 cells, 2 stations each. Cell ids include 18 (hard-coded depot default in train/validate/benchmark).
    cell_ids = [18, 19, 20, 21]
    layout = {18: (0, 0), 19: (0, 1), 20: (1, 0), 21: (1, 1)}  # (row, col)
    nodes = {}
    g = nx.MultiDiGraph(crs="EPSG:4326")
    nid = 1001
    cell_nodes = {}
    for cid in cell_ids:
        r, c = layout[cid]
        cell_nodes[cid] = []
        for k in range(2):
            lat = 42.35 + 0.004 * (1 - r) + 0.001 * k
            lon = -71.10 + 0.004 * c + 0.001 * k
            g.add_node(nid, y=lat, x=lon, street_count=1)
            nodes[nid] = (lat, lon)
            cell_nodes[cid].append(nid)
            nid += 1
    ids = list(nodes)
    for a in ids:
        for b in ids:
            if a != b:
                g.add_edge(a, b, length=100.0)
    ox.save_graphml(g, os.path.join(root, "utils", "network.graphml"))

    # cells
    pos = {v: k for k, v in layout.items()}
    cells = {}
    for cid in cell_ids:
        r, c = layout[cid]
        cell = Cell(cid, Polygon([(0, 0), (1, 0), (1, 1)]), 300)
        cell._nodes = list(cell_nodes[cid])
        cell._center_node = cell_nodes[cid][0]
        cell._adjacent_cells = {
            "up": pos.get((r - 1, c)),
            "down": pos.get((r + 1, c)),
            "left": pos.get((r, c - 1)),
            "right": pos.get((r, c + 1)),
        }
        cells[cid] = cell
    with open(os.path.join(root, "utils", "cell_data.pkl"), "wb") as f:
        pickle.dump(cells, f)

    # distance matrix (rows: node_id; columns: str(node_id))
    rng = np.random.default_rng(0)
    rows = []
    for a in ids:
        row = {"node_id": a}
        for b in ids:
            row[str(b)] = 0 if a == b else int(300 + 50 * abs(a - b))
        rows.append(row)
    pl.DataFrame(rows).write_csv(os.path.join(root, "utils", "distance_matrix.csv"))

    # nearby nodes: everything within the same cell
    nearby = {n: [m for m in cell_nodes[next(c for c in cell_ids if n in cell_nodes[c])] if m != n] for n in ids}
    with open(os.path.join(root, "utils", "nearby_nodes.pkl"), "wb") as f:
        pickle.dump(nearby, f)

    # EV matrices
    pl.DataFrame({"hour": list(range(24)), **{d: [30] * 24 for d in DAYS}}).write_csv(
        os.path.join(root, "utils", "ev_velocity_matrix.csv"))
    pl.DataFrame({"hour": list(range(24)), **{d: [0.1] * 24 for d in DAYS}}).write_csv(
        os.path.join(root, "utils", "ev_consumption_matrix.csv"))

    # global rates + PMF matrices (uniform over all (n+1)^2 pairs incl. virtual station 10000)
    all_ids = ids + [10000]
    global_rates = {}
    for d in DAYS:
        os.makedirs(os.path.join(root, "matrices", "09-10", d), exist_ok=True)
        for ts in range(8):
            global_rates[(d, ts)] = 0.01
            # real data never has external->external trips (preprocess_data filters them)
            p = 1.0 / (len(all_ids) ** 2 - 1)
            rows = [{"node_id": a, **{str(b): (0.0 if (a == 10000 and b == 10000) else p) for b in all_ids}}
                    for a in all_ids]
            pl.DataFrame(rows).write_csv(
                os.path.join(root, "matrices", "09-10", d, f"{ts:02d}-pmf-matrix.csv"))
    with open(os.path.join(root, "utils", "global_rates.pkl"), "wb") as f:
        pickle.dump(global_rates, f)


if __name__ == "__main__":
    build(sys.argv[1])
    print("synthetic data written to", sys.argv[1])
