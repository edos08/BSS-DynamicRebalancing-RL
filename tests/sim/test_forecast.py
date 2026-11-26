import datetime as dt

import numpy as np
import pytest

from bss_sim import forecast as fc
from bss_sim.config import SimConfig
from bss_sim.engine import TRIP_END, At
from bss_sim.simulator import Simulator
from bss_sim.synthetic import make_bundle

START = "2024-09-17T00:00"


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("syn"), n_side=24, n_stations=10, weekday_trips=3000, seed=4)


def make(syn, level="origin", oracle=False, **extra):
    cfg = {"fleet": {"size": 250}, "trucks": {"count": 0}, "telemetry": {"event_log": True},
           "info": {"inflight_knowledge": level, "allow_oracle": oracle}, **extra}
    sim = Simulator.from_scenario(syn, SimConfig.from_dict(cfg))
    n = sim.scenario.n_nodes
    sim.add_partition("quad", (np.arange(n) * 4 // n).astype(np.int64))
    sim.reset(seed=2, start=START)
    return sim


def test_ride_time_table_matches_the_generator(syn):
    sim = make(syn)
    m = sim.scenario.demand
    table = fc.RideTimeTable(m, samples=256)
    F = table.F
    assert F.min() >= 0 and F.max() <= 1 and (np.diff(F, axis=-1) >= -1e-7).all()
    assert (F[..., 0] < 0.05).all() and (F[..., -1] == 1).all()
    o, d, k = 2, 7, 1                                          # generator draws between these places
    n = 20000
    rng = np.random.default_rng(9)
    _, _, ride = m.nodes_and_times(np.full(n, o, np.int16), np.full(n, d, np.int16), np.full(n, k, np.int8),
                                   rng.random(n), rng.random(n), rng.random(n))
    for t in (300.0, 600.0, 1200.0):
        assert table.cdf(k, o, d, t) == pytest.approx((ride <= t).mean(), abs=0.05)
    assert table.inverse(k, o, d, float(table.cdf(k, o, d, 600.0))) == pytest.approx(600.0, abs=STEP)


STEP = fc.STEP_S


def test_mean_departures_equal_the_model_expectation(syn):
    sim = make(syn)
    sim.run(At(8 * 3600))
    f = sim.forecast(3600, "quad")
    m = sim.scenario.demand
    date = dt.date(2024, 9, 17)
    lam = m.rates[m.group(date), 8].astype(float) * m.month(date)
    W, _ = sim._fc.W("quad")
    np.testing.assert_allclose(f.departures, lam[:-1].sum(axis=1) @ W, rtol=1e-9)
    assert f.departures.shape == (4,) and f.provider == "mean" and f.inflight_knowledge == "origin"
    np.testing.assert_allclose(f.arrivals, f.arrivals_new + f.arrivals_inflight)


def test_destination_level_is_exact(syn):
    sim = make(syn, level="destination", oracle=True)
    sim.run(At(8.5 * 3600))
    t = sim.now
    sim._fc = fc.ForecastContext(sim)
    pred = fc.inflight_mean(sim._fc, 1800, "quad", "destination")
    riding = sim.inflight("destination").bike.tolist()
    sim.run(At(t + 1800))
    log = sim.event_log()
    groups = sim._parts["quad"].node_to_group
    real = np.zeros(4)
    for b in riding:   # first trip end of each bike that was riding at t
        m = (log["event"] == TRIP_END) & (log["bike"] == b) & (log["t"] > t)
        if m.any():
            real[groups[log["node"][m][0]]] += 1
    np.testing.assert_array_equal(pred, real)


@pytest.mark.parametrize("level", ["none", "count", "origin"])
def test_levels_run(syn, level):
    sim = make(syn, level=level)
    sim.run(At(8 * 3600))
    f = sim.forecast(1800, "quad")
    assert (f.arrivals_inflight >= 0).all() and f.arrivals_inflight.sum() > 0
    with pytest.raises(ValueError):
        sim.forecast(1800, "quad", provider="crystal_ball")
    with pytest.raises(KeyError):
        sim.forecast(1800, "nope")
    with pytest.raises(ValueError):
        sim.forecast(0, "quad")
