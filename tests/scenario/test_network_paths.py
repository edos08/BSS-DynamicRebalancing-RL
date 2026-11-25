import networkx as nx
import numpy as np
import polars as pl
import pytest

from bss_scenario.network import Neighbours, Projection, adjacency, all_pairs, walk_neighbours
from bss_scenario.network.graph import EDGE_SCHEMA, NODE_SCHEMA, Network


def make_net(edges, n):
    """edges: (u, v, length) directed."""
    nodes = pl.DataFrame({"node_idx": np.arange(n, dtype=np.int32), "osm_id": np.arange(n) + 100,
                          "lat": np.zeros(n), "lon": np.zeros(n), "x": np.zeros(n), "y": np.zeros(n)},
                         schema=NODE_SCHEMA)
    e = pl.DataFrame([(u, v, float(w), "residential", None, False, None) for u, v, w in edges],
                     schema=EDGE_SCHEMA, orient="row")
    return Network(nodes, e, Projection(0, 0), 0)


# ring 0 -> 1 -> 2 -> 3 -> 0 (one-way), plus a two-way shortcut 0 <-> 2
RING = [(0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100), (0, 2, 150), (2, 0, 150)]


def test_all_pairs_matches_networkx():
    net = make_net(RING, 4)
    d = all_pairs(net)
    assert d.dtype == np.float32 and d.shape == (4, 4)
    g = nx.DiGraph()
    g.add_weighted_edges_from(RING)
    ref = dict(nx.all_pairs_dijkstra_path_length(g))
    for i in range(4):
        for j in range(4):
            assert d[i, j] == pytest.approx(ref[i][j])
    assert d[1, 0] == 250  # one-way: 1 -> 2 -> 0 via shortcut
    assert d[0, 1] == 100


def test_all_pairs_custom_weight_and_unreachable():
    net = make_net(RING, 4)
    assert all_pairs(net, np.full(len(RING), 2.0))[0, 3] == 4.0  # 0 -> 2 -> 3 in hops x 2
    with pytest.raises(ValueError, match="not strongly connected"):
        all_pairs(make_net([(0, 1, 10)], 2))
    with pytest.raises(ValueError, match="positive"):
        adjacency(make_net([(0, 1, 0)], 2))


def test_walk_neighbours_undirected_sorted_and_radius():
    net = make_net(RING, 4)
    nb = walk_neighbours(net, max_radius_m=150)
    idx, dist = nb.of(1)
    # walking ignores one-way: 1 -> 0 is 100 m on foot (250 m by bike)
    assert idx.tolist() == [1, 0, 2] and dist.tolist() == [0, 100, 100]
    idx, dist = nb.of(0)
    assert idx.tolist() == [0, 1, 3, 2] and dist.tolist() == [0, 100, 100, 150]
    assert np.all(np.diff(nb.indptr) >= 1)


def test_neighbours_roundtrip(tmp_path):
    nb = walk_neighbours(make_net(RING, 4), 120)
    nb.save(tmp_path / "nb.npz")
    back = Neighbours.load(tmp_path / "nb.npz")
    assert back.max_radius_m == 120
    for a, b in ((nb.indptr, back.indptr), (nb.indices, back.indices), (nb.dist, back.dist)):
        np.testing.assert_array_equal(a, b)
