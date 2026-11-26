import numpy as np
import pytest

from bss_policies import DoNothing, StaticTSP, run_episode
from bss_sim.config import SimConfig
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle

START = "2024-09-17T05:00"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("c"), n_side=24, n_stations=12, weekday_trips=2500, seed=3)


def make(bundle, mode="docked", trucks=1, debug=True):
    cfg = {"mode": mode, "debug": debug, "fleet": {"size": 100, "initial_distribution": "uniform"},
           "trucks": {"count": trucks}, "telemetry": {"event_log": False}}
    return Simulator.from_scenario(bundle, SimConfig.from_dict(cfg))


class Recorder:
    """Wraps a controller and records what it submits."""

    def __init__(self, inner):
        self.inner, self.name = inner, inner.name
        self.submitted = []

    def describe(self):
        return self.inner.describe()

    def reset(self, sim):
        orig = sim.submit

        def submit(truck, cmds, replace=False):
            self.submitted.append((sim.now, truck, list(cmds), replace))
            return orig(truck, cmds, replace)
        sim.submit = submit
        self.inner.reset(sim)

    def act(self, sim, notes):
        self.inner.act(sim, notes)


def test_static_tsp_plans_on_schedule_and_moves_bikes(bundle):
    sim = make(bundle)
    rec = Recorder(StaticTSP(every_h=3, first_h=6))
    res = run_episode(sim, rec, seed=1, start=START, duration_s=12 * 3600)    # invariants after every event
    times = sorted({t for t, *_ in rec.submitted})
    assert times[:3] == [6 * 3600, 9 * 3600, 12 * 3600]
    assert all(r for *_, r in rec.submitted)                                  # each schedule replaces the last
    assert res.kpis["bikes_moved"] > 0 and res.kpis["truck_km"] > 0
    assert res.controller["every_h"] == 3 and res.controller["zones"] == "stations"


def test_static_tsp_respects_capacity_and_swaps(bundle):
    sim = make(bundle, mode="dockless")
    rec = Recorder(StaticTSP(every_h=2, swap_below=0.5))
    res = run_episode(sim, rec, seed=2, start=START, duration_s=8 * 3600)
    cap = sim.config.trucks.capacity
    for _, _, cmds, _ in rec.submitted:
        picks = [c.n for c in cmds if type(c).__name__ == "PickUp"]
        assert all(0 < n <= cap for n in picks)
    kinds = {type(c).__name__ for _, _, cmds, _ in rec.submitted for c in cmds}
    assert {"GoTo", "PickUp", "Drop"} <= kinds
    assert res.kpis["batteries_swapped"] >= 0


def test_static_tsp_is_reproducible_and_helps_when_docked(bundle):
    def run(c, seed):
        return run_episode(make(bundle, debug=False), c, seed=seed, start=START, duration_s=17 * 3600).kpis
    a, b = run(StaticTSP(), 4), run(StaticTSP(), 4)
    assert a == b
    base = np.mean([run(DoNothing(), s)["city_service_level"] for s in (4, 5)])
    static = np.mean([run(StaticTSP(), s)["city_service_level"] for s in (4, 5)])
    assert static > base
    assert run(DoNothing(), 4)["operating_cost_usd"] == 0.0      # an idle truck costs nothing


def test_bad_zone_kind():
    with pytest.raises(ValueError):
        StaticTSP(zones="hexagons")
    with pytest.raises(ValueError):
        StaticTSP(every_h=0)
