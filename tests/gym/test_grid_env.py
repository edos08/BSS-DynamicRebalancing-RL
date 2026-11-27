import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import bss_gym  # noqa: E402,F401
from bss_gym.grid_env import CHARGE, DOWN, DROP, LEFT, PICK_UP, RIGHT, STAY, UP, GridZoneEnv  # noqa: E402
from bss_sim.synthetic import make_bundle  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("g"), n_side=20, n_stations=8, weekday_trips=800, seed=4)


def make(bundle, **kw):
    return GridZoneEnv(bundle, sim_config={"fleet": {"size": 80}, "external": {"reservoir": 40}},
                       start_dates=("2024-09-17",), start_hour=7, episode_days=kw.pop("days", 0.25), **kw)


def test_spaces_and_gymnasium_checker(bundle):
    env = make(bundle)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and env.n_zones == 16
    assert env.edge_index.shape[0] == 2 and env.edge_attr.shape == (env.edge_index.shape[1], 1)
    assert (env.edge_attr >= 0).all()
    from gymnasium.utils.env_checker import check_env
    check_env(make(bundle), skip_render_check=True)


def test_semi_mdp_stepping(bundle):
    env = make(bundle)
    env.reset(seed=1)
    t0 = env.sim.now
    total_steps = 0
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(int(rng.integers(8)))
        assert info["steps"] >= 1 and len(info["step_rewards"]) == info["steps"] == len(info["failures"])
        assert info["elapsed_s"] == pytest.approx(30.0 * info["steps"])        # whole 30 s steps
        assert r == pytest.approx(sum(info["step_rewards"]))
        assert not term
        total_steps += info["steps"]
        done = trunc
    assert env.sim.now == pytest.approx(t0 + 0.25 * 86400) and total_steps == 0.25 * 86400 / 30
    with pytest.raises(ValueError):
        env.step(9)


def test_moves_drops_and_invalid_actions(bundle):
    env = make(bundle)
    env.reset(seed=2)
    z = env.truck_zone
    obs, r, *_, info = env.step(STAY)
    assert info["steps"] == 2 and env.truck_zone == z and not info["invalid"]     # Wait(60) = 2 steps
    missing = [a for a, flag in zip((UP, DOWN, LEFT, RIGHT), env.borders(z)) if flag]
    present = [a for a, flag in zip((UP, DOWN, LEFT, RIGHT), env.borders(z)) if not flag]
    if missing:                                         # no neighbour: invalid, one step, legacy penalty
        *_, info = env.step(missing[0])
        assert info["invalid"] and info["steps"] == 1 and info["step_rewards"][0] <= -1.0 + 1e-9
    a = present[0]
    *_, info = env.step(a)
    assert env.truck_zone == env._grid[z].neighbours[a] and info["steps"] >= 1
    load = info["truck_load"]
    *_, info = env.step(DROP)
    assert info["invalid"] or info["truck_load"] == load - 1
    env.sim.trucks[0].load.clear()                     # empty truck: drop is invalid
    for b in np.flatnonzero(env.sim.fleet.status == 3):
        env.sim.fleet.set_status(int(b), 4)
        env.sim.depot_bikes.append(int(b))
    *_, info = env.step(DROP)
    assert info["invalid"]


def test_reproducible_with_seed(bundle):
    def roll(seed):
        env = make(bundle)
        env.reset(seed=seed)
        out = []
        for a in [UP, RIGHT, PICK_UP, DROP, CHARGE, STAY, DOWN, LEFT] * 3:
            obs, r, *_, info = env.step(a)
            out.append((r, info["steps"], obs["agent"].sum()))
        return out
    assert roll(3) == roll(3)
    assert roll(3) != roll(4)


def test_critic_and_failure_penalty(bundle):
    env = make(bundle, days=0.5)
    env.reset(seed=5)
    fails = 0
    pen = []
    done = False
    while not done:
        *_, trunc, info = env.step(STAY)
        fails += sum(info["failures"])
        pen += [r for r, f in zip(info["step_rewards"][1:], info["failures"][1:])]
        done = trunc
    assert fails > 0 and all(p <= 0 for p in pen)
    assert np.isfinite(env.critic).all() and env.critical.dtype == bool
    assert env.global_critic >= 0


def test_binary_critic_as_legacy(bundle):
    """Default critic is the legacy binary one: critical zones score 1, global = their count, and the
    shaped reward stays on the legacy scale (the raw exp score made it reach +-10^4)."""
    env = make(bundle)
    assert env.binary_critic
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, r, _, truncated, _ = env.step(int(rng.integers(8)))
        crit = env.critic
        assert np.all((crit == 1.0) | (crit <= 0.0))
        assert np.array_equal(env.critical, env.raw_critic > 0.0)
        assert env.global_critic == float(env.critical.sum()) <= env.n_zones
        if truncated:
            break
    raw = make(bundle, binary_critic=False)
    raw.reset(seed=3)
    assert raw.global_critic == pytest.approx(float(np.maximum(raw.raw_critic, 0.0).sum()))
