import networkx as nx
import numpy as np
import polars as pl
import pytest

from bss_scenario import cli
from bss_scenario.traffic import ODSampleConfig, build_od_sample, endpoint_nodes, sample_od_pairs
from bss_scenario.traffic.sample import OD_SCHEMA, haversine_m


def grid_graph(n=12, step_deg=0.004):
    """n x n street grid around Cambridge; one column is a busway, one edge a trunk link."""
    g = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            g.add_node(f"{i}_{j}", y=str(42.36 + i * step_deg), x=str(-71.13 + j * step_deg))
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] < n and b[1] < n:
                    hw = "busway" if j == n - 1 and dj == 0 else "residential"
                    g.add_edge(f"{a[0]}_{a[1]}", f"{b[0]}_{b[1]}", highway=hw)
                    g.add_edge(f"{b[0]}_{b[1]}", f"{a[0]}_{a[1]}", highway=hw)
    g.add_node("link", y="42.40", x="-71.09")
    g.add_edge("0_0", "link", highway="trunk_link")
    g.add_edge("1_1", "2_2", highway="['residential', 'unclassified']")
    return g


def test_endpoints_exclude_links_and_busways():
    nodes = endpoint_nodes(grid_graph())
    assert "link" not in nodes["node"].to_list()
    assert nodes.height == 12 * 12  # busway column still touches residential rows
    assert nodes["node"].is_sorted()


def test_sample_is_stratified_and_deterministic():
    nodes = endpoint_nodes(grid_graph())
    cfg = ODSampleConfig(n_pairs=40, bands_m=((300, 1000), (1000, 2000)), grid=(2, 2), seed=1)
    a = sample_od_pairs(nodes, cfg)
    b = sample_od_pairs(nodes, cfg)
    assert a.equals(b)
    assert dict(a.schema) == OD_SCHEMA
    assert a.group_by("band").len().sort("band")["len"].to_list() == [20, 20]
    for band, (lo, hi) in enumerate(cfg.bands_m):
        d = a.filter(pl.col("band") == band)["crow_m"]
        assert d.min() >= lo and d.max() < hi
    assert a["o_cell"].n_unique() == 4  # every area used as origin
    assert a.select(pl.struct("o_node", "d_node").is_unique().all()).item()
    np.testing.assert_allclose(a["crow_m"], haversine_m(a["o_lat"], a["o_lon"], a["d_lat"], a["d_lon"]))
    assert not a.equals(sample_od_pairs(nodes, ODSampleConfig(n_pairs=40, bands_m=cfg.bands_m, grid=(2, 2), seed=2)))


def test_impossible_band_raises():
    nodes = endpoint_nodes(grid_graph(n=3))
    with pytest.raises(ValueError, match="cannot find a pair"):
        sample_od_pairs(nodes, ODSampleConfig(n_pairs=1, bands_m=((50_000, 60_000),), max_tries=20))


def test_cli(tmp_path, capsys):
    path = tmp_path / "g.graphml"
    nx.write_graphml(grid_graph(), path)
    out = tmp_path / "traffic" / "od.parquet"
    assert cli.main(["traffic", "sample", "--graph", str(path), "--out", str(out), "--n-pairs", "8"]) == 0
    assert pl.read_parquet(out).height == 8
    assert "8 pairs written" in capsys.readouterr().out
    assert build_od_sample(path, tmp_path / "again.parquet", ODSampleConfig(n_pairs=8)).equals(pl.read_parquet(out))
