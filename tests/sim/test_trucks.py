import pytest

from bss_sim.commands import GoTo, ToDepot, Wait
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle

START = "2024-09-17T06:00"


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=8, weekday_trips=400, seed=2)


def make(bundle, **trucks):
    cfg = {"fleet": {"size": 60}, "external": {"reservoir": 40}, "debug": True,
           "trucks": {"count": 2, "initial_load": 5, **trucks}}
    return Simulator.from_scenario(bundle, SimConfig.from_dict(cfg))


def notes_of(notes, kind):
    return [n for n in notes if n.kind == kind]


def test_reset_reports_idle_trucks_at_the_depot(syn):
    sim = make(syn)
    notes = sim.reset(seed=1, start=START)
    assert [(n.kind, n.data["truck"]) for n in notes] == [("truck_idle", 0), ("truck_idle", 1)]
    for tr in sim.trucks:
        assert tr.stop == sim.depot_stop and tr.drive_node == sim.depot_drive and len(tr.load) == 5
        assert sim.scenario.bike_to_drive[sim.depot_stop] == sim.depot_drive
    assert sim.fleet.counts()["on_truck"] == 10


def test_goto_takes_the_travel_time_and_moves_the_stop(syn):
    sim = make(syn)
    sim.reset(seed=1, start=START)
    node = 333
    dest = int(sim.scenario.bike_to_drive[node])
    expected = sim.travel.time(sim.depot_drive, dest, sim.now)
    t0 = sim.now
    sim.submit(0, [GoTo(node)])
    assert sim.trucks[0].busy and sim.trucks[0].eta == pytest.approx(t0 + expected)
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(7200))
    done = notes_of(notes, "command_done")
    assert len(done) == 1 and done[0].data["result"].ok and done[0].t == pytest.approx(t0 + expected)
    assert notes_of(notes, "truck_idle")[0].data == {"truck": 0, "reason": "queue_empty"}
    tr = sim.trucks[0]
    assert (tr.stop, tr.drive_node) == (node, dest) and not tr.busy
    assert tr.km == pytest.approx(sim.travel.distance(sim.depot_drive, dest) / 1000) and tr.drive_s == pytest.approx(expected)
    assert sim.now == pytest.approx(t0 + expected)  # stopped right at the idle notification


def test_queue_order_results_and_invalid_commands(syn):
    sim = make(syn)
    sim.reset(seed=1, start=START)
    t0 = sim.now
    sim.submit(0, [GoTo(50), Wait(120), GoTo(-1), Wait(-5), GoTo(10**9), ToDepot()])
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(4 * 3600))
    res = [n.data["result"] for n in notes_of(notes, "command_done")]
    assert [type(r.command).__name__ for r in res] == ["GoTo", "Wait", "GoTo", "Wait", "GoTo", "ToDepot"]
    assert [r.ok for r in res] == [True, True, False, False, False, True]
    assert all(r.reason.startswith("invalid") for r in res[2:5])
    assert res[1].t_end - res[1].t_start == pytest.approx(120) and res[0].t_start == t0
    assert all(a.t_end <= b.t_start + 1e-9 for a, b in zip(res, res[1:]))
    assert sim.trucks[0].stop == sim.depot_stop
    assert len(notes_of(notes, "truck_idle")) == 1


def test_replace_cancels_the_queue_but_not_the_current_command(syn):
    sim = make(syn)
    sim.reset(seed=1, start=START)
    sim.submit(0, [Wait(600), GoTo(10), GoTo(20)])
    sim.submit(0, [GoTo(30)], replace=True)
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(7200))
    res = [n.data["result"] for n in notes_of(notes, "command_done")]
    assert [(type(r.command).__name__, r.reason) for r in res] == [
        ("GoTo", "cancelled"), ("GoTo", "cancelled"), ("Wait", ""), ("GoTo", "")]
    assert sim.trucks[0].stop == 30


def test_trucks_are_independent_and_runs_reproducible(syn):
    def run(seed, start="depot"):
        sim = make(syn, start=start)
        sim.reset(seed=seed, start=START)
        sim.submit(0, [GoTo(100), Wait(300), GoTo(5)])
        sim.submit(1, [GoTo(200)])
        notes = sim.run(TimeLimit(3 * 3600))
        return [(n.t, n.kind, n.data["truck"]) for n in notes], [(t.stop, t.km) for t in sim.trucks]
    a = run(1)
    assert a == run(1)
    idle = [x for x in a[0] if x[1] == "truck_idle"]
    assert {x[2] for x in idle} == {0, 1}
    assert a[1][0][0] == 5 and a[1][1][0] == 200
    r1, r2 = run(3, "random"), run(4, "random")
    assert r1 == run(3, "random") and r1 != r2


def test_travel_noise_changes_drive_times_reproducibly(syn):
    def eta(seed):
        sim = Simulator.from_scenario(syn, SimConfig.from_dict({"fleet": {"size": 60}, "travel": {"noise_sigma": 0.3}}))
        sim.reset(seed=seed, start=START)
        sim.submit(0, [GoTo(333)])
        return sim.trucks[0].eta
    assert eta(1) == eta(1) and eta(1) != eta(2)


def test_no_trucks(bundle):
    sim = Simulator.from_scenario(bundle, SimConfig.from_dict({"fleet": {"size": 20}, "trucks": {"count": 0}}))
    assert sim.reset(seed=1, start=START) == []
    sim.run(TimeLimit(3600))
    assert sim.fleet.counts()["on_truck"] == 0


def test_run_returns_at_once_when_a_stop_notification_is_pending(syn):
    sim = make(syn)
    sim.reset(seed=1, start=START)
    t = sim.now
    sim.submit(0, [GoTo(-1)])                 # invalid: done at once, the truck is idle again
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(3600))
    assert sim.now == t and [n.kind for n in notes] == ["command_done", "truck_idle"]
    notes = sim.run(TimeLimit(60))            # a time-only condition is not affected
    assert sim.now == t + 60
