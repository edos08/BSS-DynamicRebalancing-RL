import numpy as np
import pytest

from bss_sim.config import SimConfig
from bss_sim.engine import REDIRECT, TRIP_END, TRIP_REQUEST, TRIP_START, OnNotification, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.users import EXTERNAL_EMPTY

START = "2024-09-17T06:00"


def cfg(**kw):
    """Small fleet, invariants checked after every event; ``kw`` sections are merged into the defaults."""
    base = {"fleet": {"size": 20}, "external": {"reservoir": 30}, "debug": True}
    for k, v in kw.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return SimConfig.from_dict(base)


@pytest.fixture(scope="module")
def sim(bundle):
    return Simulator.from_scenario(bundle, cfg(telemetry={"event_log": True}))


def test_two_days_with_invariants_checked_after_every_event(sim):
    sim.reset(seed=1, start=START)
    assert sim.now == 6 * 3600
    sim.run(TimeLimit(2 * 86400))
    assert sim.now == 6 * 3600 + 2 * 86400
    k = sim.kpis()
    assert k["requests"] > 100
    assert k["requests"] == k["served"] + k["failed"] + k["skipped_external_both"]
    c = sim.fleet.counts()
    assert sum(c.values()) == 50
    in_flight = c["reserved"] + c["riding"]
    assert k["served"] == k["trips_ended"] + k["to_external"] + in_flight
    assert 0 < k["service_level"] <= 1


def test_same_seed_same_log_whatever_ran_before(sim):
    def log(seed):
        sim.reset(seed=seed, start=START)
        sim.run(TimeLimit(86400))
        return sim.event_log()
    a = log(3)
    log(4)
    log(5)
    b = log(3)
    assert len(a["t"]) > 50
    for col in a:
        assert a[col].tobytes() == b[col].tobytes(), col
    c = log(4)
    assert len(c["t"]) != len(a["t"]) or not np.array_equal(c["bike"], a["bike"])


def test_log_is_time_ordered_and_consistent(sim):
    sim.reset(seed=2, start=START)
    sim.run(TimeLimit(86400))
    log = sim.event_log()
    assert (np.diff(log["t"]) >= 0).all()
    assert log["t"][0] >= 6 * 3600
    served = log["request"][(log["event"] == TRIP_REQUEST) & (log["outcome"] == 0)]
    starts = log["request"][log["event"] == TRIP_START]
    ends = log["request"][log["event"] == TRIP_END]
    assert set(starts) <= set(served) and set(ends) <= set(served)


def test_run_in_pieces_equals_one_run(sim):
    sim.reset(seed=6, start=START)
    sim.run(TimeLimit(43200))
    one = sim.event_log()
    sim.reset(seed=6, start=START)
    for _ in range(12):
        sim.run(TimeLimit(3600))
    pieces = sim.event_log()
    for col in one:
        np.testing.assert_array_equal(one[col], pieces[col])


def test_wakeups_and_stop_conditions(sim):
    sim.reset(seed=1, start=START)
    sim.wakeup(after=600, tag="a")
    sim.wakeup(at=sim.now + 900, tag="b")
    notes = sim.run(OnNotification("wakeup") | TimeLimit(3600))
    assert [(n.kind, n.data["tag"], n.t) for n in notes] == [("wakeup", "a", 6 * 3600 + 600)]
    assert sim.now == 6 * 3600 + 600
    notes = sim.run(TimeLimit(3600))  # not stopping on wakeups: collected and returned
    assert [n.data["tag"] for n in notes] == ["b"] and sim.now == 6 * 3600 + 600 + 3600
    with pytest.raises(ValueError):
        sim.run(OnNotification())
    with pytest.raises(ValueError):
        sim.wakeup(after=-1)
    with pytest.raises(ValueError):
        sim.wakeup(at=1.0, after=1.0)
    assert str(sim.datetime) == "2024-09-17 07:10:00"


def test_empty_reservoir_fails_arrivals_from_outside(bundle):
    s = Simulator.from_scenario(bundle, cfg(external={"reservoir": 0}))
    s.reset(seed=1, start=START)
    s.run(TimeLimit(86400))
    assert s.kpis()["failures"]["external_empty"] > 0
    assert s.kpis_.failures[EXTERNAL_EMPTY] == s.kpis()["failures"]["external_empty"]


def test_initial_distributions(bundle):
    s = Simulator.from_scenario(bundle, cfg(fleet={"initial_distribution": "from_snapshot"}))
    nodes = np.arange(20) % 5
    s.reset(seed=1, start=START, snapshot=nodes)
    np.testing.assert_array_equal(s.fleet.node[:20], nodes)
    with pytest.raises(ValueError):
        s.reset(seed=1, start=START)
    with pytest.raises(ValueError):
        s.reset(seed=1, start=START, snapshot=nodes[:5])
    u = Simulator.from_scenario(bundle, cfg(fleet={"initial_distribution": "uniform"}))
    u.reset(seed=1, start=START)
    weights = u.scenario.departure_weights()
    assert (weights[u.fleet.node[:20]] > 0).all()


def test_replay_generator_and_config_checks(bundle):
    s = Simulator.from_scenario(bundle, cfg(demand={"generator": "replay"}))
    s.reset(seed=1, start=START)
    s.run(TimeLimit(86400))
    assert s.kpis()["requests"] > 0
    with pytest.raises(ValueError, match="walk_radius_m"):
        Simulator.from_scenario(bundle, cfg(users={"walk_radius_m": 5000}))


def docked(bundle, **kw):
    return Simulator.from_scenario(bundle, cfg(mode="docked", telemetry={"event_log": True}, **kw))


def test_docked_mode_keeps_bikes_at_stations_within_capacity(bundle):
    s = docked(bundle, fleet={"size": 40})  # 3 stations x 15 docks: nearly full
    cap = s.scenario.node_capacity()
    assert cap.sum() == 45
    s.reset(seed=1, start=START)
    assert (s.fleet.n_bikes <= cap).all() and s.fleet.n_bikes[cap == 0].sum() == 0
    s.run(TimeLimit(2 * 86400))  # invariants (incl. capacity) checked after every event
    k = s.kpis()
    assert k["redirects"] > 0 and k["redirect_km"] > 0
    log = s.event_log()
    assert (log["event"] == REDIRECT).sum() == k["redirects"]


def test_redirect_radius_and_overflow(bundle):
    s = docked(bundle, fleet={"size": 40}, users={"redirect_radius_m": 0})
    s.reset(seed=1, start=START)
    s.run(TimeLimit(86400))
    k = s.kpis()
    assert k["no_dock"] == k["redirects"] > 0  # every ride on to another dock is beyond 0 m
    assert "no_dock" not in k["failures"]       # a return problem, not a failed request
    full = docked(bundle, fleet={"size": 45}, external={"reservoir": 200})
    full.reset(seed=2, start=START)
    full.run(TimeLimit(86400))
    assert full.kpis()["dock_overflow"] > 0     # all docks taken: bikes left over capacity, still counted


def test_docked_uniform_fills_by_capacity(bundle):
    s = docked(bundle, fleet={"size": 30, "initial_distribution": "uniform"})
    s.reset(seed=3, start=START)
    cap = s.scenario.node_capacity()
    assert sorted(s.fleet.n_bikes[cap > 0].tolist()) != [0, 15, 15]  # spread, not packed
    assert (s.fleet.n_bikes <= cap).all() and s.fleet.n_bikes.sum() == 30


def test_docked_needs_enough_docks(bundle):
    with pytest.raises(ValueError, match="docks"):
        docked(bundle, fleet={"size": 46}).reset(seed=1, start=START)
