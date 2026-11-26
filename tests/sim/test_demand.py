import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scenario"))

from bss_sim.calendar import day_group as sim_day_group  # noqa: E402
from bss_sim.demand import DemandModel, PoissonDemand, RideTimes, TripRequests  # noqa: E402


def test_calendar_matches_scenario_package():
    from bss_scenario.demand.calendar import day_group
    day = dt.date(2023, 1, 1)
    for _ in range(800):
        for scheme in ("per_weekday", "day_types", "workday_weekend"):
            assert sim_day_group(day, scheme) == day_group(day, scheme)
        day += dt.timedelta(days=1)


def test_hour_is_reproducible_and_independent_of_order(model):
    a = PoissonDemand(model, seed=7)
    b = PoissonDemand(model, seed=7)
    day = dt.date(2024, 9, 17)
    b.hour(day, 3)  # generating something else first must not change anything
    x, y = a.hour(day, 8), b.hour(day, 8)
    for f in ("t_s", "o_place", "d_place", "o_node", "d_node", "kind"):
        np.testing.assert_array_equal(getattr(x, f), getattr(y, f))
    np.testing.assert_array_equal(np.isnan(x.ride_time_s), np.isnan(y.ride_time_s))
    assert len(PoissonDemand(model, seed=8).hour(day, 8)) != len(x) or not np.array_equal(
        PoissonDemand(model, seed=8).hour(day, 8).t_s, x.t_s)


def test_requests_are_consistent(model):
    gen = PoissonDemand(model, seed=1)
    r = gen.day(dt.date(2024, 9, 18))
    ext = model.external
    assert len(r) > 0 and np.all(np.diff(r.t_s) >= 0) and r.t_s.min() >= 0 and r.t_s.max() < 86400
    assert np.all((r.o_node == -1) == (r.o_place == ext)) and np.all((r.d_node == -1) == (r.d_place == ext))
    inside = (r.o_place != ext) & (r.d_place != ext)
    assert np.all(np.isfinite(r.ride_time_s[inside])) and np.all(r.ride_time_s[inside] > 0)
    assert np.all(np.isnan(r.ride_time_s[~inside]))
    assert not np.any((r.o_place == ext) & (r.d_place == ext))


def test_counts_match_expectation(model):
    gen = PoissonDemand(model, seed=3)
    day = dt.date(2024, 9, 18)
    n = np.array([len(PoissonDemand(model, seed=s).hour(day, 8)) for s in range(300)])
    exp = np.array([PoissonDemand(model, seed=s).expected(day, 8) for s in range(300)])
    assert n.mean() == pytest.approx(exp.mean(), rel=0.05)
    assert gen.day_factor(day) == gen.day_factor(day)


def test_ride_times_match_scenario_model(bundle):
    from bss_scenario.demand.ride_time import RideTimeModel

    ref = RideTimeModel.load(bundle / "demand" / "ride_time.npz")
    sim = RideTimes(bundle / "demand" / "ride_time.npz")
    dist = np.array([0.0, 300.0, 1200.0, 2500.0])
    kind = np.array([0, 1, 0, 1])
    for t in (60.0, 300.0, 900.0):
        np.testing.assert_allclose(sim.cdf(t, dist, kind), ref.cdf(t, dist, kind))


def test_docked_mode_uses_station_nodes(bundle):
    m = DemandModel.load(bundle, spread_mode="station")
    r = PoissonDemand(m, seed=2).day(dt.date(2024, 9, 18))
    inside = r.o_node >= 0
    assert set(np.unique(r.o_node[inside])) <= set(m.spreader.station_nodes.tolist())


def test_empty_concat():
    assert len(TripRequests.concat([])) == 0


def test_sim_never_imports_builder_or_legacy():
    import ast

    import bss_sim
    forbidden = {"bss_scenario", "gymnasium_env", "preprocessing", "rl_training", "benchmark", "torch", "polars"}
    for path in Path(bss_sim.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                assert name.split(".")[0] not in forbidden, f"{path.name} imports {name}"


def test_replay_reproduces_recorded_trips(bundle, model):
    from bss_sim.demand import ReplayDemand

    rep = ReplayDemand(model, bundle, seed=4)
    day = dt.date(2024, 9, 18)
    r = rep.day(day)
    assert len(r) == 60                                   # 60 recorded trips per day in the fixture
    assert np.all(np.diff(r.t_s) >= 0)
    hours = set((r.t_s // 3600).astype(int).tolist())
    assert hours <= {8, 9, 17, 18}
    ext = model.external
    inside = (r.o_place != ext) & (r.d_place != ext)
    np.testing.assert_allclose(r.ride_time_s[inside], 480.0)  # recorded 8-minute trips
    assert np.all(np.isnan(r.ride_time_s[~inside]))
    assert np.all((r.o_node == -1) == (r.o_place == ext))
    again = ReplayDemand(model, bundle, seed=4).hour(day, 8)
    np.testing.assert_array_equal(again.o_node, rep.hour(day, 8).o_node)
    assert r.kind.mean() == pytest.approx(0.2, abs=0.1)
    with pytest.raises(KeyError, match="no recorded trips"):
        rep.day(dt.date(2023, 1, 5))


def test_replay_times_are_exact(bundle, model):
    """Regression: minute * 60 overflowed Int8 in the export and shifted trips into the previous hour."""
    import polars as pl

    from bss_sim.demand import ReplayDemand

    day = dt.date(2024, 9, 18)
    t = pl.read_parquet(bundle / "trips" / "2024-09.parquet").filter("valid")
    local = t.filter(pl.col("start_time").dt.date() == day)["start_time"]
    expected = np.sort((local.dt.hour().cast(pl.Float64) * 3600 + local.dt.minute().cast(pl.Float64) * 60
                        + local.dt.second().cast(pl.Float64)).to_numpy())
    np.testing.assert_allclose(ReplayDemand(model, bundle, seed=0).day(day).t_s, expected)


def test_vectorised_ride_time_sampling_matches_the_loop(bundle):
    rt = RideTimes(bundle / "demand" / "ride_time.npz")
    rng = np.random.default_rng(1)
    n = 5000
    dist = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(1.0, 9000.0, n))
    kind = rng.integers(0, 2, n)
    u = rng.random(n)
    np.testing.assert_allclose(rt.sample(dist, kind, u), rt._sample_loop(dist, kind, u), rtol=1e-12)
