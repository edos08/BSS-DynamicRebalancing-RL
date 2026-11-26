import numpy as np
import pytest

from bss_sim.commands import GoTo, Restock, SwapBattery, ToDepot, Wait
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.state import ELECTRIC
from bss_sim.synthetic import make_bundle

START = "2024-09-17T06:00"
STOP, OTHER = 210, 150


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=8, weekday_trips=400, seed=2)


def make(syn, nodes, **trucks):
    """No users; all city bikes electric at ``nodes`` with low charge; one truck at STOP."""
    cfg = {"demand": {"generator": "none"}, "debug": True, "external": {"reservoir": 0},
           "fleet": {"size": len(nodes), "initial_distribution": "from_snapshot", "ebike_share": 1.0,
                     "initial_battery": [0.05, 0.3]},
           "trucks": {"count": 1, "initial_load": 0, **trucks}}
    sim = Simulator.from_scenario(syn, SimConfig.from_dict(cfg))
    sim.reset(seed=1, start=START, snapshot=np.array(nodes))
    sim.submit(0, [GoTo(STOP)])
    sim.run(OnNotification("truck_idle") | TimeLimit(7200))
    return sim


def go(sim, *cmds, until="truck_idle"):
    sim.submit(0, list(cmds))
    return sim.run(OnNotification(until) | TimeLimit(12 * 3600))


def results(notes):
    return [n.data["result"] for n in notes if n.kind == "command_done"]


def test_swap_lowest_charge_first(syn):
    sim = make(syn, [STOP] * 6, battery_stock_initial=10)
    f = sim.fleet
    low = sorted(f.bikes_at(STOP), key=lambda b: f._batt[b])[:4]
    (res,) = results(go(sim, SwapBattery(4)))
    assert res.ok and res.executed == 4 and res.t_end - res.t_start == pytest.approx(60 + 4 * 60)
    assert all(f._batt[b] == sim.battery.range_km for b in low)
    tr = sim.trucks[0]
    assert (tr.stock, tr.used, tr.swapped) == (6, 4, 4)
    (res,) = results(go(sim, SwapBattery(5)))
    assert (res.executed, res.reason) == (2, "not_enough_ebikes")   # only 2 not full yet


def test_resume_mode_goes_to_the_depot_and_comes_back(syn):
    sim = make(syn, [STOP] * 6 + [OTHER] * 3, battery_stock_initial=2)
    notes = go(sim, SwapBattery(6), GoTo(OTHER), SwapBattery(3))
    kinds = [n.kind for n in notes]
    assert kinds.count("truck_restocking") == 1 and kinds.count("truck_resumed") == 1
    res = results(notes)
    names = [(type(r.command).__name__, r.executed, r.reason) for r in res]
    assert names == [("SwapBattery", 2, "stock_empty"), ("ToDepot", 1, ""), ("Restock", 0, ""), ("GoTo", 1, ""),
                     ("SwapBattery", 4, ""), ("GoTo", 1, ""), ("SwapBattery", 3, "")]
    assert res[3].command == GoTo(STOP)            # came back to the interrupted stop
    f = sim.fleet
    assert all(f._batt[b] == sim.battery.range_km for b in range(9))
    assert sim.trucks[0].stock == 60 - 7 and sim.trucks[0].used == 7


def test_resume_before_driving_to_a_swap(syn):
    sim = make(syn, [OTHER] * 2, battery_stock_initial=0)
    notes = go(sim, GoTo(OTHER), SwapBattery(2))
    res = results(notes)
    assert [type(r.command).__name__ for r in res] == ["ToDepot", "Restock", "GoTo", "SwapBattery"]
    assert res[-1].ok and res[2].command == GoTo(OTHER)   # heads where it was going after restocking
    assert [n.kind for n in notes].count("truck_restocking") == 1


def test_idle_mode_stops_and_reports(syn):
    sim = make(syn, [STOP] * 5, battery_stock_initial=2, on_empty="idle")
    notes = go(sim, SwapBattery(5), Wait(60), GoTo(OTHER))
    res = results(notes)
    assert [(type(r.command).__name__, r.reason) for r in res] == [
        ("SwapBattery", "stock_empty"), ("Wait", "cancelled"), ("GoTo", "cancelled")]
    idle = [n for n in notes if n.kind == "truck_idle"]
    assert len(idle) == 1 and idle[0].data["reason"] == "stock_empty"
    notes = go(sim, SwapBattery(1))
    assert results(notes)[0].reason == "stock_empty" and notes[-1].data["reason"] == "stock_empty"
    notes = go(sim, ToDepot(), Restock(), GoTo(STOP), SwapBattery(3))
    assert results(notes)[-1].ok and "truck_restocking" not in [n.kind for n in notes]


def test_swap_needs_ebikes_within_radius(syn):
    sim = make(syn, [OTHER] * 3)
    (res,) = results(go(sim, SwapBattery(2, radius_m=50)))
    assert (res.executed, res.reason) == (0, "not_enough_ebikes") and res.t_start == res.t_end
    assert all(sim.fleet.kind_l[b] == ELECTRIC for b in range(3))
