import numpy as np
import pytest

from bss_sim import bench
from bss_sim.config import SimConfig
from bss_sim.engine import TimeLimit
from bss_sim.scenario import Scenario
from bss_sim.simulator import Simulator
from bss_sim.synthetic import _erfinv, make_bundle


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=12, weekday_trips=1200, seed=3)


def test_files_and_structure(small):
    sc = Scenario.load(small)
    assert sc.n_nodes == 400 and len(sc.station_node) == 12 and sc.demand.n_places == 13
    ip, idx, d = sc.walk_indptr, sc.walk_indices, sc.walk_dist
    for v in (0, 57, 399):  # sorted by distance, self first, within the max radius
        a, b = ip[v], ip[v + 1]
        assert idx[a] == v and d[a] == 0 and (np.diff(d[a:b]) >= 0).all() and d[b - 1] <= sc.walk_max_m
    assert sc.bike_dist[0, 21] == 120.0  # one block east and one north: 2 x 60 m
    sp = sc.demand.spreader
    for s in range(sp.n):
        assert sp.probs[sp.indptr[s]:sp.indptr[s + 1]].sum() == pytest.approx(1.0)
    assert not list(small.rglob("*.parquet")) and not list(small.rglob("*.pkl"))


def test_demand_scale_and_external_share(small):
    sc = Scenario.load(small)
    r = sc.demand.rates.astype(float)
    assert r[0].sum() == pytest.approx(1200, rel=1e-4)            # weekday total
    assert r[3].sum() < r[0].sum()                                 # Sundays quieter
    ext = r[0, :, -1, :].sum() + r[0, :, :, -1].sum()
    assert ext / r[0].sum() == pytest.approx(0.54, abs=0.01)
    assert r[0, 17].sum() > 5 * r[0, 3].sum()                       # evening peak vs night


def test_runs_with_invariants(small):
    sim = Simulator.from_scenario(small, SimConfig.from_dict({"fleet": {"size": 100}, "debug": True}))
    sim.reset(seed=0, start="2024-09-16T06:00")
    sim.run(TimeLimit(86400))
    assert sim.kpis()["requests"] > 500


def test_bench_reports_budget_fields(small, tmp_path):
    res = bench.run(small, repeat=1, threads=2)
    for key in ("load_s", "week_s", "peak_mb", "week_docked_s", "week_partition_s", "threads_speedup"):
        assert res[key] > 0
    assert set(res["budget"]) == {"load_s", "week_s", "peak_mb"}
    out = tmp_path / "b.json"
    assert bench.main([str(small), "--repeat", "1", "--threads", "1", "--json", str(out)]) == 0
    assert out.exists()


def test_erfinv():
    from math import erf
    y = np.linspace(-0.999, 0.999, 41)
    assert max(abs(erf(v) - u) for v, u in zip(_erfinv(y), y)) < 1e-6
