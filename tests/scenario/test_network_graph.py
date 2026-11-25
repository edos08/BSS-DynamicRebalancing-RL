import networkx as nx
import numpy as np
import pytest

from bss_scenario.network import Projection, load_graphml, main_highway
from bss_scenario.traffic.sample import haversine_m


def write_graph(path):
    g = nx.MultiDiGraph(crs="epsg:4326")
    coords = {"30": (42.3700, -71.1100), "10": (42.3710, -71.1100), "20": (42.3710, -71.1085), "40": (42.3720, -71.1085)}
    for n, (lat, lon) in coords.items():
        g.add_node(n, y=str(lat), x=str(lon))
    # strongly connected triangle 10-20-30 (two-way), 40 only reachable (one-way in)
    for a, b, length in (("10", "20", 120.0), ("20", "30", 150.0), ("30", "10", 111.0)):
        g.add_edge(a, b, length=str(length), highway="residential", oneway="False", maxspeed="20 mph", name="A St")
        g.add_edge(b, a, length=str(length), highway="residential", oneway="False", maxspeed="20 mph", name="A St")
    g.add_edge("10", "20", length="200.0", highway="['secondary', 'residential']", oneway="True",
               maxspeed="['25 mph', '20 mph']", name="['B St', 'C St']")  # longer parallel edge: dropped
    g.add_edge("20", "40", length="100.0", highway="primary", oneway="True")
    g.add_edge("10", "10", length="5.0", highway="residential", oneway="False")  # self loop: dropped
    nx.write_graphml(g, path)
    return path


def test_load_graphml(tmp_path):
    net = load_graphml(write_graph(tmp_path / "g.graphml"))
    assert net.dropped_nodes == 1 and net.n == 3
    assert net.nodes["osm_id"].to_list() == [10, 20, 30]
    assert net.nodes["node_idx"].to_list() == [0, 1, 2]
    assert net.edges.height == 6  # one row per directed pair, no self loop
    e = net.edges.filter((net.edges["u"] == 0) & (net.edges["v"] == 1)).row(0, named=True)
    assert e["length_m"] == 120.0 and e["highway"] == "residential" and e["oneway"] is False


def test_main_highway_and_multi_values():
    assert main_highway("['residential', 'secondary']") == "secondary"
    assert main_highway("busway") == "busway"
    assert main_highway(None) == "unclassified"


def test_projection_matches_haversine():
    p = Projection(42.37, -71.11)
    lat = np.array([42.36, 42.39]); lon = np.array([-71.13, -71.08])
    x, y = p.to_xy(lat, lon)
    planar = np.hypot(x[1] - x[0], y[1] - y[0])
    assert planar == pytest.approx(haversine_m(lat[0], lon[0], lat[1], lon[1]), rel=1e-3)


def test_shared_projection(tmp_path):
    p = Projection(42.0, -71.0)
    net = load_graphml(write_graph(tmp_path / "g.graphml"), projection=p)
    assert net.projection == p
    x, y = p.to_xy(net.nodes["lat"], net.nodes["lon"])
    np.testing.assert_allclose(net.nodes["x"], x)
