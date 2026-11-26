import datetime as dt

import numpy as np
import polars as pl
import pytest

from bss_scenario.demand import Places
from bss_scenario.demand.shape import ShapeConfig, ShapeModel, count_tensor, exposure, fit_shape, smooth_rates

TZ = "America/New_York"
PLACES = Places(("A", "B"))  # A=0, B=1, EXTERNAL=2


def trips(rows):
    return pl.DataFrame({
        "start_time": pl.Series([dt.datetime(*t) for t, *_ in rows]).dt.replace_time_zone(TZ),
        "o_place": [o for _, o, _ in rows], "d_place": [d for *_, d in rows],
    })


def test_count_tensor_and_exposure():
    tue, sat = (2024, 7, 9, 8, 15), (2024, 7, 13, 17, 0)
    t = trips([(tue, 0, 1), (tue, 0, 1), (sat, 1, 2)])
    N = count_tensor(t, PLACES, "workday_weekend")
    assert N.shape == (2, 24, 3, 3) and N.sum() == 3
    assert N[0, 8, 0, 1] == 2 and N[1, 17, 1, 2] == 1
    days = [dt.date(2024, 7, 8) + dt.timedelta(days=i) for i in range(7)]
    np.testing.assert_array_equal(exposure(days, None, "workday_weekend"), [5, 2])


def test_no_smoothing_gives_raw_rates():
    rng = np.random.default_rng(0)
    N = rng.poisson(3.0, size=(2, 24, 3, 3))
    N[:, :, 2, 2] = 0
    E = np.array([10.0, 4.0])
    r = smooth_rates(N, E, alpha=0.0, beta=0.0, external=2)
    np.testing.assert_allclose(r, N / E[:, None, None, None], rtol=1e-6)


def test_smoothing_preserves_origin_daily_totals_and_masks_external():
    rng = np.random.default_rng(1)
    N = rng.poisson(0.3, size=(2, 24, 3, 3))
    N[:, :, 2, 2] = 0
    E = np.array([10.0, 4.0])
    r = smooth_rates(N, E, alpha=4.0, beta=8.0, external=2).astype(np.float64)
    assert np.all(r[:, :, 2, 2] == 0)
    np.testing.assert_allclose(r.sum(axis=(1, 3)) * E[:, None], N.sum(axis=(1, 3)), rtol=1e-5)
    # a pair never observed in an hour still gets a small positive rate if the origin is active
    assert r[0][N[0] == 0].min() >= 0 and (r[0][(N[0] == 0)] > 0).any()


def test_rare_pair_is_pulled_towards_pattern():
    # origin A: 100 trips to B at 8h and one odd round trip A->A at 9h (same morning band)
    N = np.zeros((1, 24, 3, 3))
    N[0, 8, 0, 1] = 100
    N[0, 9, 0, 0] = 1
    E = np.array([10.0])
    raw = smooth_rates(N, E, 0.0, 0.0, 2)
    smooth = smooth_rates(N, E, 4.0, 8.0, 2)
    assert smooth[0, 9, 0, 0] < 0.5 * raw[0, 9, 0, 0]    # the single odd trip is shrunk towards A->B
    assert smooth[0, 8, 0, 1] == pytest.approx(raw[0, 8, 0, 1], rel=0.1)  # the big flow barely moves


def test_fit_and_roundtrip(tmp_path):
    days = [dt.date(2024, 7, 8) + dt.timedelta(days=i) for i in range(7)]
    t = trips([((2024, 7, 9, 8, 0), 0, 1), ((2024, 7, 10, 9, 0), 1, 2), ((2024, 7, 13, 12, 0), 2, 0)])
    m = fit_shape(t, PLACES, days, None, ShapeConfig("day_types", alpha=1.0, beta=1.0))
    assert m.rates.shape == (4, 24, 3, 3) and m.groups == ("mon-thu", "fri", "sat", "sun")
    assert m.counts_total == 3
    m.save(tmp_path / "shape.npz")
    back = ShapeModel.load(tmp_path / "shape.npz")
    np.testing.assert_array_equal(back.rates, m.rates)
    assert back.places == PLACES and back.config == m.config and back.groups == m.groups


def test_bad_config():
    with pytest.raises(ValueError):
        ShapeConfig(alpha=-1)
    with pytest.raises(ValueError):
        ShapeConfig(scheme="monthly")
