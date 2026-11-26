import gymnasium as gym
import numpy as np
import pytest

import gymnasium_env  # noqa: F401  (registers the environments)

RESET = dict(total_timeslots=2, maximum_number_of_bikes=40, minimum_number_of_bikes=1, depot_id=18)


@pytest.fixture()
def dyn_env(data_path, tmp_path):
    env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path=data_path,
                   results_path=str(tmp_path), seed=1)
    yield env
    env.close()


@pytest.fixture()
def static_env(data_path, tmp_path):
    env = gym.make("gymnasium_env/StaticEnv-v0", data_path=data_path,
                   results_path=str(tmp_path), seed=1)
    yield env
    env.close()


@pytest.mark.parametrize("mode", ["poisson", "oracle"])
def test_dynamic_episode_and_critic_scores(dyn_env, mode):
    obs, info = dyn_env.reset(options=dict(RESET, critic_mode=mode))
    n_cells = len(info["cell_dict"])
    assert obs.shape == dyn_env.observation_space.shape == (17 + 2 * n_cells,)

    cells = info["cell_dict"]
    prev = {cid: c.get_critic_score() for cid, c in cells.items()}
    done, steps = False, 0
    while not done:
        obs, reward, done, _, info = dyn_env.step(dyn_env.action_space.sample())
        steps += 1
        assert np.isfinite(reward)
        for cid, c in cells.items():
            # regression: old_critic_score used to be snapshotted twice per update, so it held the
            # raw score of the *current* step instead of the final score of the previous one
            assert c.get_old_critic_score() == prev[cid]
            prev[cid] = c.get_critic_score()
            assert c.get_critic_score() <= 1.0 + 1e-9
            if c.is_critical():
                assert c.get_critic_score() > 0.0
    assert steps > 0


@pytest.mark.parametrize("start", [1, 6, 23])
def test_static_starting_rebalancing_hour(static_env, start):
    # regression: the option was read but the default hour was used to build the schedule
    static_env.reset(options=dict(RESET, num_rebalancing_events=2, starting_rebalancing_event=start))
    assert static_env.unwrapped._rebalancing_hours == sorted((start + i) % 24 for i in (0, 12))


def test_static_episode_runs(static_env):
    static_env.reset(options=dict(RESET, total_timeslots=8, num_rebalancing_events=2))
    done = False
    while not done:
        _, _, done, _, info = static_env.step(0)
    assert isinstance(info["failures"], int) and info["rebalance_times"]


# ---------------------------------------------------------------------------
# Critic forecast (critic_mode)
# ---------------------------------------------------------------------------

def _critics(env):
    env.unwrapped._update_cells_metrics()
    return {cid: c.get_critic_score() for cid, c in env.unwrapped._cells.items()}


def test_invalid_critic_mode(dyn_env):
    with pytest.raises(ValueError):
        dyn_env.reset(options=dict(RESET, critic_mode="nope"))


def test_poisson_critic_does_not_look_at_the_simulated_future(dyn_env):
    """No leak: emptying the realised event buffer must not change a poisson critic,
    while it does change the oracle one."""
    unwrapped = dyn_env.unwrapped

    dyn_env.reset(options=dict(RESET, critic_mode="poisson"))
    before = _critics(dyn_env)
    unwrapped._event_buffer.clear()
    assert _critics(dyn_env) == before

    dyn_env.reset(options=dict(RESET, critic_mode="oracle"))
    before = _critics(dyn_env)
    unwrapped._event_buffer.clear()
    assert _critics(dyn_env) != before


def test_poisson_forecast_is_seeded(data_path, tmp_path):
    def forecast(seed):
        env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path=data_path, results_path=str(tmp_path))
        try:
            env.reset(seed=seed, options=RESET)
            return [t.tolist() for t in env.unwrapped._forecast_times]
        finally:
            env.close()

    assert forecast(5) == forecast(5)
    assert forecast(5) != forecast(6)


def test_forecast_matches_the_rates(dyn_env):
    """Sampled forecast must be a Poisson process with the simulator's rates."""
    env = dyn_env.unwrapped
    dyn_env.reset(options=RESET)
    dep, arr = env._slot_cell_rates("monday", 0)
    duration = 3 * 3600
    n = 300
    totals = np.zeros(2)
    for _ in range(n):
        samples = env._sample_forecast_slot("monday", 0, 0.0)
        totals += [sum((d > 0).sum() for _, d, _ in samples), sum((d < 0).sum() for _, d, _ in samples)]
    expected = np.array([dep.sum(), arr.sum()]) * duration
    sd = np.sqrt(expected / n)                      # sd of the mean of n Poisson counts
    assert np.all(np.abs(totals / n - expected) < 4 * sd)


def test_forecast_window_slides_with_the_slots(dyn_env):
    """After a slot change the drawn scenario of the upcoming slot becomes the current one."""
    env = dyn_env.unwrapped
    dyn_env.reset(options=dict(RESET, total_timeslots=3))
    duration = 3 * 3600
    upcoming = [t[(t >= duration)] - duration for t in env._forecast_times]
    done = False
    while not done and env._timeslots_completed == 0:
        _, _, done, _, _ = dyn_env.step(0)
    now = [t[t < duration - 100] for t in env._forecast_times]
    # events of the former "next slot" are now in [0, duration) (those not yet in the past)
    for before, after in zip(upcoming, now):
        assert set(np.round(after, 6)) <= set(np.round(before, 6))


def test_inflight_trips_are_known(dyn_env):
    """Arrivals of trips already on the road are used exactly as the simulator will deliver them."""
    env = dyn_env.unwrapped
    dyn_env.reset(options=dict(RESET, total_timeslots=3, critic_mode="poisson"))
    checked = 0
    for _ in range(120):
        dyn_env.step(0)
        truth = {i: [] for i in range(len(env._cell_order))}
        for e in env._event_buffer:
            if e.is_arrival():
                trip = e.get_trip()
                end = trip.get_end_location()
                # departed (a bike was assigned), not failed, ends inside a cell
                if trip.get_bike() is not None and not trip.is_failed() and end.get_station_id() != 10000:
                    truth[env._cell_index[end.get_cell().get_id()]].append(e.time)
        for i in truth:
            tracked = sorted(t for t in env._inflight_arrivals[i] if t >= env._env_time)
            real = sorted(t for t in truth[i] if t >= env._env_time)
            assert tracked == real
            checked += len(real)
    assert checked > 0   # the test actually saw trips on the road


def test_forecast_does_not_count_arrivals_of_departed_trips(dyn_env):
    """A forecast arrival only counts while its departure is still in the future."""
    env = dyn_env.unwrapped
    dyn_env.reset(options=dict(RESET, critic_mode="poisson"))
    # let (simulated) time pass without touching the forecast scenario
    env._env_time = 3000
    for i in range(len(env._cell_order)):
        env._inflight_arrivals[i] = []
    mx, aft = env._flows_from_forecast()
    # recompute by brute force: only trips whose departure is >= now, i.e. valid_from >= now
    for i, cell_id in enumerate(env._cell_order):
        t, d, v = env._forecast_times[i], env._forecast_deltas[i], env._forecast_valid_from[i]
        sel = (t >= 3000) & (t <= 3000 + 3 * 3600) & (v >= 3000)
        net = np.cumsum(d[sel][np.argsort(t[sel], kind="stable")], dtype=int)
        peak = max(0, int(net.max())) if net.size else 0
        assert mx.get(cell_id, 0) == peak
