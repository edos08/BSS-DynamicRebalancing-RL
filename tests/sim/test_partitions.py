import numpy as np
import pytest

from bss_sim.config import SimConfig
from bss_sim.engine import TRIP_END, TRIP_START, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.users import REASONS

START = "2024-09-17T06:00"


@pytest.fixture(scope="module")
def sim(bundle):
    cfg = SimConfig.from_dict({"fleet": {"size": 25}, "external": {"reservoir": 30}, "debug": True,
                               "telemetry": {"event_log": True}})
    return Simulator.from_scenario(bundle, cfg)


def test_counters_match_a_recount_and_the_log(sim):
    n = sim.scenario.n_nodes
    grid = np.arange(n) % 7
    partial = np.where(np.arange(n) % 3 == 0, -1, np.arange(n) % 4)  # some nodes in no group
    sim.add_partition("grid", grid)
    sim.add_partition("partial", partial)
    with pytest.raises(ValueError):
        sim.add_partition("grid", grid)
    sim.reset(seed=5, start=START)
    sim.run(TimeLimit(2 * 86400))  # debug: bikes and e-bike bands recounted after every event
    c = sim.partition("grid")
    k = sim.kpis()
    assert c.bikes.sum() == sim.fleet.counts()["parked"]
    assert c.requests.sum() == k["city_requests"]
    failures = c.failures.sum(axis=0)
    for r, name in enumerate(REASONS):
        if name in k["failures"]:
            assert failures[r] == k["failures"][name], name
    log = sim.event_log()
    starts = log["node"][log["event"] == TRIP_START]
    ends = log["node"][log["event"] == TRIP_END]
    np.testing.assert_array_equal(c.departures, np.bincount(grid[starts], minlength=7))
    np.testing.assert_array_equal(c.arrivals, np.bincount(grid[ends], minlength=7))
    p = sim.partition("partial")
    assert p.bikes.shape == (4,)
    assert p.bikes.sum() == (partial[sim.fleet.node[sim.fleet.status == 0]] >= 0).sum()


def test_counters_restart_at_reset_and_follow_battery_changes(sim):
    sim.reset(seed=6, start=START)
    c = sim.partition("grid")
    assert c.requests.sum() == 0 and c.departures.sum() == 0
    f = sim.fleet
    e = np.flatnonzero((f.status == 0) & (f.kind == 1))
    if len(e):
        before = sim.partition("grid").ebikes.sum(axis=0)
        f.set_battery(int(e[0]), 0.0)
        after = sim.partition("grid").ebikes.sum(axis=0)
        assert after[0] >= before[0] and after.sum() == before.sum()
    sim.check_invariants()


def test_partition_added_mid_run(sim):
    sim.reset(seed=7, start=START)
    sim.run(TimeLimit(43200))
    sim.add_partition("late", np.zeros(sim.scenario.n_nodes, dtype=np.int64))
    c = sim.partition("late")
    assert c.bikes[0] == sim.fleet.counts()["parked"] and c.requests[0] == 0
    sim.run(TimeLimit(3600))
    sim.check_invariants()


def test_bad_partitions_rejected(sim):
    n = sim.scenario.n_nodes
    with pytest.raises(ValueError):
        sim.add_partition("short", np.zeros(n - 1, dtype=int))
    with pytest.raises(ValueError):
        sim.add_partition("float", np.zeros(n))
    with pytest.raises(ValueError):
        sim.add_partition("neg", np.full(n, -2))
