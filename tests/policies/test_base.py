import ast
import pathlib

import pytest

from bss_policies import Controller, DoNothing, run_episode
from bss_sim.commands import GoTo
from bss_sim.config import SimConfig
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle

START = "2024-09-17T06:00"


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    b = make_bundle(tmp_path_factory.mktemp("p"), n_side=20, n_stations=8, weekday_trips=800, seed=1)
    return Simulator.from_scenario(b, SimConfig.from_dict({"fleet": {"size": 80}}))


def test_do_nothing_runs_and_reports(sim):
    res = run_episode(sim, DoNothing(), seed=1, start=START, duration_s=6 * 3600)
    assert res.kpis["requests"] > 50 and res.kpis["truck_km"] == 0
    assert res.controller == {"name": "do_nothing"} and res.decisions == 0 and res.seed == 1
    assert res.metadata["seed"] == 1 and sim.now == 12 * 3600


class Shuttle(Controller):
    """Goes back and forth between two nodes whenever idle."""
    name = "shuttle"

    def reset(self, sim):
        self.turns = 0
        self.seen = []

    def act(self, sim, notes):
        self.seen += [n.kind for n in notes]
        if any(n.kind == "truck_idle" for n in notes):
            self.turns += 1
            sim.submit(0, [GoTo(5 if self.turns % 2 else 300)])


def test_controller_acts_on_idle_trucks(sim):
    c = Shuttle()
    res = run_episode(sim, c, seed=2, start=START, duration_s=3 * 3600)
    assert res.decisions > 5 and c.turns > 5 and res.kpis["truck_km"] > 0
    assert "command_done" in c.seen
    again = run_episode(sim, Shuttle(), seed=2, start=START, duration_s=3 * 3600)
    assert again.kpis == res.kpis                         # reproducible


class Stuck(Controller):
    name = "stuck"

    def act(self, sim, notes):
        sim.submit(0, [GoTo(-1)])                         # invalid: completes at once, truck idle again


def test_a_controller_that_stops_the_clock_is_caught(sim):
    with pytest.raises(RuntimeError, match="did not move"):
        run_episode(sim, Stuck(), seed=1, start=START, duration_s=3600, max_stalls=50)


def test_bss_sim_never_imports_policies():
    root = pathlib.Path(__file__).parents[2] / "src" / "bss_rebalancing" / "sim" / "src" / "bss_sim"
    for f in root.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith("bss_policies") for n in names), f
