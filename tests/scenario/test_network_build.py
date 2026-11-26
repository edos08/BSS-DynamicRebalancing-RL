import json

import networkx as nx
import numpy as np
import polars as pl

from bss_scenario import cli
from bss_scenario.network import Neighbours, load_graphml, nearest_nodes


def street_grid(path, n=6, step=0.002, busway_col=None, oneway_row=None):
    g = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            g.add_node(str(1000 + i * n + j), y=str(42.37 + i * step), x=str(-71.12 + j * step))
    def nid(i, j):
        return str(1000 + i * n + j)
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < n and j + dj < n:
                    hw = "busway" if busway_col is not None and dj == 0 and j == busway_col else "residential"
                    length = str(step * 111_000 * (0.74 if dj else 1.0))
                    g.add_edge(nid(i, j), nid(i + di, j + dj), length=length, highway=hw, maxspeed="20 mph", oneway="False")
                    if not (oneway_row is not None and di == 0 and i == oneway_row):
                        g.add_edge(nid(i + di, j + dj), nid(i, j), length=length, highway=hw, maxspeed="20 mph", oneway="False")
    nx.write_graphml(g, path)
    return path


def test_busway_excluded_before_scc(tmp_path):
    p = street_grid(tmp_path / "d.graphml", busway_col=0)
    net = load_graphml(p, exclude_highways={"busway"})
    assert "busway" not in net.edges["highway"].to_list()
    assert net.n == 36  # rows still connect every node


def test_nearest_nodes_requires_same_projection(tmp_path):
    p = street_grid(tmp_path / "g.graphml")
    a = load_graphml(p)
    idx, dist = nearest_nodes(a, a)
    assert np.array_equal(idx, np.arange(a.n)) and np.all(dist == 0)
    b = load_graphml(p, projection=type(a.projection)(0.0, 0.0))
    try:
        nearest_nodes(a, b)
    except ValueError as exc:
        assert "projection" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_networks(tmp_path, capsys):
    bike = street_grid(tmp_path / "bike.graphml", n=6, step=0.001)
    drive = street_grid(tmp_path / "drive.graphml", n=4, step=0.002, busway_col=1, oneway_row=1)
    out = tmp_path / "network"
    assert cli.main(["networks", "--bike", str(bike), "--drive", str(drive), "--out", str(out),
                     "--walk-radius-max", "300"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["bike_dist.npy", "bike_edges.parquet", "bike_nodes.parquet", "bike_to_drive.npz",
                     "bike_walk_nbrs.npz", "drive_dist.npy", "drive_edges.parquet", "drive_nodes.npz",
                     "drive_nodes.parquet", "drive_time.npy", "drive_time_by_class.npy", "network_report.json"]
    bd = np.load(out / "bike_dist.npy")
    dt = np.load(out / "drive_time.npy")
    assert bd.shape == (36, 36) and dt.shape == (16, 16) and bd.dtype == dt.dtype == np.float32
    assert np.all(np.diag(dt) == 0) and np.all(dt[~np.eye(16, dtype=bool)] > 0)
    edges = pl.read_parquet(out / "drive_edges.parquet")
    assert {"speed_kmh", "freeflow_s"} <= set(edges.columns) and "busway" not in edges["highway"].to_list()
    # one-way row: travel times are asymmetric somewhere
    assert np.abs(dt - dt.T).max() > 0
    by_class = np.load(out / "drive_time_by_class.npy")
    assert by_class.shape == (3, 16, 16)
    np.testing.assert_allclose(by_class.sum(axis=0), dt, rtol=1e-5, atol=1e-3)
    dd = np.load(out / "drive_dist.npy")
    assert dd.shape == (16, 16) and np.all(np.diag(dd) == 0) and (dd >= dt * 20 / 3.6 * 0.999).all()  # 20 mph
    with np.load(out / "drive_nodes.npz") as z:
        assert z["lat"].shape == (16,) and z["road_classes"].tolist() == ["major", "secondary", "local"]
    b2d = np.load(out / "bike_to_drive.npz")
    assert b2d["node"].shape == (36,) and b2d["node"].max() < 16
    nb = Neighbours.load(out / "bike_walk_nbrs.npz")
    assert nb.max_radius_m == 300 and nb.dist.max() <= 300
    report = json.loads((out / "network_report.json").read_text())
    assert report["bike"]["nodes"] == 36 and report["drive"]["excluded_highways"] == ["busway"]
    assert "bike_dist.npy" in report["files_mb"]
