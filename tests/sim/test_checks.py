import datetime as dt

import numpy as np
import pytest

from bss_sim.checks import compare, jsd_bits, ks, tvd
from bss_sim.demand import PoissonDemand, ReplayDemand, TripRequests


def test_metric_properties():
    a = np.array([1.0, 2.0, 3.0])
    assert tvd(a, 2 * a) == 0 and jsd_bits(a, a) == 0
    assert tvd(np.array([1, 0]), np.array([0, 1])) == 1
    assert jsd_bits(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert ks(x, x) == 0 and ks(x, x + 20) == pytest.approx(1.0)
    assert np.isnan(ks(np.array([]), x))


def test_compare_identical_sets():
    r = TripRequests(np.array([3600.0, 7300.0]), np.array([0, 1], np.int16), np.array([1, 2], np.int16),
                     np.array([5, 6], np.int32), np.array([7, -1], np.int32), np.array([300.0, np.nan], np.float32),
                     np.array([0, 1], np.int8))
    m = compare(r, r, n_places=3)
    assert m["daily_mean_ratio"] == 1 and m["hourly_tvd"] == 0 and m["od_jsd"] == 0 and m["ride_ks"] == 0


def test_generator_close_to_recorded_fixture(bundle, model):
    days = [dt.date(2024, 9, d) for d in range(2, 28) if dt.date(2024, 9, d).weekday() < 4]
    rec = TripRequests.concat([ReplayDemand(model, bundle, 0).day(d) for d in days])
    gen = TripRequests.concat([PoissonDemand(model, seed=1).day(d) for d in days])
    m = compare(gen, rec, model.n_places, len(days), len(days))
    assert m["daily_mean_ratio"] == pytest.approx(1.0, abs=0.25)
    assert m["od_jsd"] < 0.05 and m["origin_tvd"] < 0.1 and m["hourly_tvd"] < 0.15
