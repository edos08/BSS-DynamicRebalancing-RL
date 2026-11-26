import numpy as np
import pytest

from bss_sim.commands import Drop, GoTo, PickUp, SwapBattery, ToDepot, Wait
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle

START = "2024-09-17T06:00"


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=20, n_stations=8, weekday_trips=400, seed=2)


def make(syn, **over):
    cfg = {"demand": {"generator": "none"}, "debug": True, "external": {"reservoir": 0},
           "fleet": {"size": 10, "initial_distribution": "from_snapshot", "ebike_share": 1.0,
                     "initial_battery": [0.1, 0.2]},
           "trucks": {"count": 1, "initial_load": 4}}
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v}
    sim = Simulator.from_scenario(syn, SimConfig.from_dict(cfg))
    sim.reset(seed=1, start=START, snapshot=np.full(10, 210))
    return sim


def test_truck_kpis_add_up(syn):
    sim = make(syn)
    sim.submit(0, [GoTo(210), PickUp(3), Drop(2), SwapBattery(2), Wait(300), ToDepot()])
    sim.run(OnNotification("truck_idle") | TimeLimit(6 * 3600))
    sim.run(TimeLimit(1800))                        # then 30 min idle
    t = sim.truck_kpis(0)
    k = sim.kpis()
    elapsed = sim.now - 6 * 3600
    assert t["drive_s"] + t["service_s"] + t["wait_s"] + t["idle_s"] == pytest.approx(elapsed)
    assert t["idle_s"] == pytest.approx(1800, abs=1e-6)
    assert (t["picked_up"], t["dropped"], t["batteries_swapped"], t["commands"]) == (3, 2, 2, 6)
    assert t["service_s"] == pytest.approx((60 + 90) + (60 + 60) + (60 + 120))
    assert t["wait_s"] == 300
    d = sim.travel
    km = (d.distance(sim.depot_drive, sim.scenario.bike_to_drive[210]) * 2) / 1000
    assert t["km"] == pytest.approx(km)
    van = sim.van
    expected = (van.service_kwh(t["service_s"] + 300)
                + van.drive_kwh(km * 500, t["drive_s"] / 2, 4) + van.drive_kwh(km * 500, t["drive_s"] / 2, 5))
    assert t["kwh"] == pytest.approx(expected, rel=1e-6)
    assert t["kwh_since_depot"] == 0.0               # back at the depot
    hours = elapsed / 3600
    assert t["cost_usd"]["driver"] == pytest.approx(hours * 22.68)
    assert t["kgco2e"] == pytest.approx(t["kwh"] / 0.9 * 0.254)
    assert k["truck_km"] == t["km"] and k["bikes_moved"] == 2 and k["batteries_swapped"] == 2
    assert k["operating_cost_usd"] == pytest.approx(t["cost_usd"]["total"]) and k["energy_flags"] == 0


def test_idle_time_excludes_the_rest_of_the_current_command(syn):
    sim = make(syn)
    sim.submit(0, [Wait(3600)])
    sim.run(TimeLimit(600))
    t = sim.truck_kpis(0)
    assert t["idle_s"] == pytest.approx(0.0, abs=1e-6)


def test_energy_flag_when_a_van_exceeds_its_battery(syn):
    sim = make(syn, vehicle={"usable_kwh": 0.05})
    sim.submit(0, [GoTo(399), GoTo(0), GoTo(399)])
    sim.run(OnNotification("truck_idle") | TimeLimit(6 * 3600))
    assert sim.truck_kpis(0)["energy_flags"] == 1    # counted once per stint between depot visits
    sim.submit(0, [ToDepot(), GoTo(0)])   # the depot snaps to the corner near node 399 here
    sim.run(OnNotification("truck_idle") | TimeLimit(6 * 3600))
    assert sim.truck_kpis(0)["energy_flags"] == 2
