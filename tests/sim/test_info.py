import numpy as np
import pytest

from bss_sim.commands import GoTo
from bss_sim.config import ConfigError, SimConfig
from bss_sim.engine import TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.state import PARKED, RIDING

START = "2024-09-17T07:00"


def make(bundle, **info):
    cfg = {"fleet": {"size": 25}, "external": {"reservoir": 30}, "telemetry": {"event_log": True}}
    if info:
        cfg["info"] = info
    return Simulator.from_scenario(bundle, SimConfig.from_dict(cfg))


def run_to_busy(sim, seed=3):
    """Run until some trips are in progress (morning peak of the test bundle)."""
    sim.reset(seed=seed, start=START)
    for _ in range(400):
        sim.run(TimeLimit(30))
        if (sim.fleet.status == RIDING).sum() >= 2:
            return
    pytest.fail("no trips in progress")


def test_origin_level_is_the_default(bundle):
    sim = make(bundle)
    run_to_busy(sim)
    obs = sim.observe()
    fl = obs.inflight
    riding = np.flatnonzero(sim.fleet.status == RIDING)
    assert fl.count == len(riding) and fl.bike.tolist() == riding.tolist()
    assert fl.d_node is None and fl.end_t is None
    ext = sim.scenario.demand.external
    assert ((fl.o_place >= 0) & (fl.o_place <= ext)).all() and (fl.start_t <= sim.now).all()
    assert ((fl.o_node == -1) == (fl.o_place == ext)).all()
    log = sim.event_log()
    for b, node, place in zip(fl.bike, fl.o_node, fl.o_place):
        if place != ext:   # the start node is where the bike was taken (last TripStart of that bike)
            starts = log["node"][(log["event"] == 2) & (log["bike"] == b)]
            assert starts[-1] == node


def test_count_and_none_levels(bundle):
    sim = make(bundle, inflight_knowledge="count")
    run_to_busy(sim)
    fl = sim.observe().inflight
    assert fl.count == (sim.fleet.status == RIDING).sum() and fl.o_place is None
    with pytest.raises(PermissionError):
        sim.inflight("origin")
    sim = make(bundle, inflight_knowledge="none")
    run_to_busy(sim)
    assert sim.observe().inflight is None


def test_destination_is_an_oracle(bundle):
    with pytest.raises(ConfigError, match="allow_oracle"):
        SimConfig.from_dict({"info": {"inflight_knowledge": "destination"}})
    sim = make(bundle, inflight_knowledge="destination", allow_oracle=True)
    run_to_busy(sim)
    assert sim.metadata()["oracle_used"] is False
    fl = sim.observe().inflight
    assert sim.metadata()["oracle_used"] is True and sim.metadata()["oracle_allowed"] is True
    assert (fl.end_t > sim.now).all()
    i = int(np.argmin(fl.end_t))
    b, node, t = int(fl.bike[i]), int(fl.d_node[i]), float(fl.end_t[i])
    sim.run(TimeLimit(t - sim.now + 1e-6))
    assert sim.fleet.status[b] == PARKED and sim.fleet.node[b] == node   # dockless: arrives where announced


def test_observation_is_read_only_and_does_not_change_the_run(bundle):
    a = make(bundle)
    a.reset(seed=5, start=START)
    a.run(TimeLimit(4 * 3600))
    b = make(bundle)
    b.reset(seed=5, start=START)
    for _ in range(16):
        b.run(TimeLimit(900))
        obs = b.observe()
        with pytest.raises(ValueError):
            obs.bike_node[0] = 3
        obs.bikes[:] = 0                         # a copy: harmless
    for col, arr in a.event_log().items():
        np.testing.assert_array_equal(arr, b.event_log()[col])
    assert b.observe().bikes.sum() == (b.fleet.status == PARKED).sum()


def test_truck_views_and_partitions(bundle):
    sim = make(bundle)
    sim.add_partition("all", np.zeros(sim.scenario.n_nodes, dtype=np.int64))
    sim.reset(seed=1, start=START)
    sim.submit(0, [GoTo(5)])
    obs = sim.observe()
    (tv,) = obs.trucks
    assert tv.busy and tv.current == GoTo(5) and tv.eta > obs.t and tv.load == 15 and tv.stock == 60
    assert obs.partitions["all"].bikes[0] == obs.bikes.sum()
    assert obs.kpis["requests"] == 0 and obs.datetime.hour == 7
