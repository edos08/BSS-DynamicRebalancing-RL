import numpy as np
import pytest

from bss_sim.commands import Drop, GoTo, PickUp, Restock, ToDepot
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.state import DEPOT, ELECTRIC, ON_TRUCK, PARKED, RESERVED
from bss_sim.synthetic import make_bundle

START = "2024-09-17T06:00"
STOP = 210           # a bike node in the middle of the 20 x 20 synthetic grid
FAR = 213            # 3 blocks (180 m) away


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=8, weekday_trips=400, seed=2)


def sim_at(syn, nodes, mode="dockless", stop=STOP, **trucks):
    """No users; city bikes at ``nodes``; one truck already at ``stop``."""
    cfg = {"mode": mode, "demand": {"generator": "none"}, "debug": True, "external": {"reservoir": 0},
           "fleet": {"size": len(nodes), "initial_distribution": "from_snapshot", "ebike_share": 0.5},
           "trucks": {"count": 1, **trucks}}
    sim = Simulator.from_scenario(syn, SimConfig.from_dict(cfg))
    sim.reset(seed=1, start=START, snapshot=np.array(nodes))
    sim.submit(0, [GoTo(stop)])
    sim.run(OnNotification("truck_idle") | TimeLimit(7200))
    assert sim.trucks[0].stop == stop
    return sim


def run_cmd(sim, *cmds):
    sim.submit(0, list(cmds))
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(6 * 3600))
    return [n.data["result"] for n in notes if n.kind == "command_done"]


def test_pickup_lowest_battery_first_with_honest_time(syn):
    sim = sim_at(syn, [STOP] * 8 + [FAR] * 4, initial_load=0)
    f = sim.fleet
    at_stop = f.bikes_at(STOP)
    ebikes = sorted((f._batt[b], b) for b in at_stop if f.kind_l[b] == ELECTRIC)
    t0 = sim.now
    (res,) = run_cmd(sim, PickUp(3, radius_m=100))
    assert res.ok and res.executed == 3
    assert res.t_end - res.t_start == pytest.approx(60 + 3 * 30)
    expected = [b for _, b in ebikes][:3]
    if len(expected) < 3:   # then classic bikes
        expected += [b for b in at_stop if f.kind_l[b] != ELECTRIC][:3 - len(expected)]
    assert sorted(sim.trucks[0].load) == sorted(expected)
    assert all(f.status[b] == ON_TRUCK for b in expected) and f.n_bikes[STOP] == 5
    assert res.t_start == t0


def test_pickup_radius_capacity_and_shortfall(syn):
    sim = sim_at(syn, [FAR] * 6, initial_load=0)
    (res,) = run_cmd(sim, PickUp(2, policy="nearest", radius_m=100))
    assert (res.executed, res.reason) == (0, "not_enough_bikes") and res.t_end == res.t_start
    (res,) = run_cmd(sim, PickUp(10, policy="nearest", radius_m=200))
    assert (res.executed, res.reason) == (6, "not_enough_bikes")
    sim = sim_at(syn, [STOP] * 6, initial_load=0, capacity=4)
    (res,) = run_cmd(sim, PickUp(6))
    assert (res.executed, res.reason) == (4, "truck_full")
    (res,) = run_cmd(sim, PickUp(1))
    assert (res.executed, res.reason) == (0, "truck_full")
    (res,) = run_cmd(sim, PickUp(1, radius_m=5000))
    assert res.reason.startswith("invalid")


def test_bikes_taken_during_the_service_are_reported(syn):
    sim = sim_at(syn, [STOP] * 3, initial_load=0)
    sim.submit(0, [PickUp(3)])
    b = sim.fleet.bikes_at(STOP)[0]
    sim.fleet.unpark(b, RESERVED)          # a user takes one while the crew is loading
    notes = sim.run(OnNotification("truck_idle") | TimeLimit(3600))
    (res,) = [n.data["result"] for n in notes if n.kind == "command_done"]
    assert (res.executed, res.reason) == (2, "taken_meanwhile")


def test_drop_here_and_spread(syn):
    sim = sim_at(syn, [], initial_load=6)
    f, tr = sim.fleet, sim.trucks[0]
    charged = sorted(tr.load, key=lambda b: (f.kind_l[b] != ELECTRIC, -f._batt[b]))
    (res,) = run_cmd(sim, Drop(2))
    assert res.ok and res.t_end - res.t_start == pytest.approx(60 + 2 * 30)
    assert sorted(f.bikes_at(STOP)) == sorted(charged[:2])
    (res,) = run_cmd(sim, Drop(3, at=(STOP, FAR)))
    assert res.ok and f.n_bikes[STOP] == 4 and f.n_bikes[FAR] == 1
    (res,) = run_cmd(sim, Drop(5))
    assert (res.executed, res.reason) == (1, "not_enough_bikes_on_truck")
    (res,) = run_cmd(sim, Drop(1))
    assert (res.executed, res.reason) == (0, "not_enough_bikes_on_truck")
    (res,) = run_cmd(sim, Drop(1, at=(399,)))
    assert res.reason.startswith("invalid")     # far corner: out of reach of the stop


def test_docked_drop_limited_by_free_docks(syn):
    sc = Simulator.from_scenario(syn, SimConfig.from_dict({"mode": "docked"})).scenario
    station = int(sc.station_node[0])
    cap = int(sc.node_capacity()[station])
    sim = sim_at(syn, [station] * (cap - 2), mode="docked", stop=station, initial_load=10)
    (res,) = run_cmd(sim, Drop(5))
    assert (res.executed, res.reason) == (2, "no_docks") and sim.fleet.n_bikes[station] == cap
    (res,) = run_cmd(sim, Drop(1))
    assert (res.executed, res.reason) == (0, "no_docks")


def test_restock_at_the_depot(syn):
    sim = sim_at(syn, [STOP] * 4, initial_load=5, depot_spare_bikes=3, battery_stock_initial=10)
    tr, f = sim.trucks[0], sim.fleet
    (res,) = run_cmd(sim, Restock())
    assert res.reason == "not_at_depot"
    res = run_cmd(sim, PickUp(4), ToDepot(), Restock(load=2))
    assert [r.ok for r in res] == [True, True, True] and res[2].executed == 7
    assert res[2].t_end - res[2].t_start == pytest.approx(600)
    assert len(tr.load) == 2 and len(sim.depot_bikes) == 3 + 7 and tr.stock == 60
    assert all(f._batt[b] == sim.battery.range_km for b in tr.load if f.kind_l[b] == ELECTRIC)
    (res,) = run_cmd(sim, Restock(load=30, batteries=5))
    assert (res.executed, res.reason) == (10, "depot_empty") and len(tr.load) == 12 and tr.stock == 5
    assert f.counts()["depot"] == 0 and (f.status == DEPOT).sum() == 0
    assert (f.status == PARKED).sum() == 0 and f.counts()["on_truck"] == 12
