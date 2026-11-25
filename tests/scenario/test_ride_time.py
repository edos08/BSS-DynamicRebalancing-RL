import numpy as np
import polars as pl
import pytest

from bss_scenario.demand.ride_time import RideTimeModel, fit_ride_time, summary

# 3 in-city places on a line, 1 km apart; EXTERNAL = 3
DIST = np.array([[0, 1000, 2000], [1000, 0, 1000], [2000, 1000, 0]], dtype=np.float32)
PLACE_NODE = np.array([0, 1, 2])


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    o = rng.integers(0, 3, n)
    d = rng.integers(0, 3, n)
    kind = rng.integers(0, 2, n)
    speed = np.where(kind == 1, 5.0, 3.5) * rng.lognormal(0, 0.15, n)  # m/s
    dist = DIST[o, d]
    dur = np.where(o == d, rng.exponential(1200, n) + 60, dist / speed)
    return pl.DataFrame({"o_place": o, "d_place": d, "duration_s": dur,
                         "bike_type": np.where(kind == 1, "electric", "classic")})


@pytest.fixture(scope="module")
def model():
    return fit_ride_time(synthetic(), PLACE_NODE, DIST, external=3, bands_m=(0, 1500, np.inf), min_trips=50)


def test_fit_recovers_speeds(model):
    med = model.speed_q[:, :, 50]
    np.testing.assert_allclose(med, [[3.5, 3.5], [5.0, 5.0]], rtol=0.03)
    assert model.counts.sum() + model.round_counts.sum() == 6000


def test_sample_and_cdf_agree(model):
    rng = np.random.default_rng(1)
    dist = np.full(20000, 2000.0)
    kind = np.ones(20000, dtype=int)
    t = model.sample(dist, kind, rng)
    assert np.median(t) == pytest.approx(400, rel=0.03)          # 2 km at 5 m/s
    for q in (200.0, 400.0, 600.0):
        assert model.cdf(q, 2000.0, 1) == pytest.approx((t <= q).mean(), abs=0.02)
    rt = model.sample(np.zeros(20000), np.zeros(20000, int), rng)
    assert np.median(rt) == pytest.approx(np.log(2) * 1200 + 60, rel=0.08)
    assert model.cdf(0.0, 2000.0, 1) == 0.0 and model.cdf(1e7, 2000.0, 1) == 1.0


def test_cdf_is_monotone(model):
    ts = np.linspace(1, 5000, 200)
    c = model.cdf(ts, 1500.0, 0)
    assert np.all(np.diff(c) >= -1e-12)


def test_sparse_cells_borrow_and_external_ignored():
    df = synthetic(400)
    df = df.with_columns(pl.when(pl.int_range(pl.len()) < 10).then(3).otherwise(pl.col("d_place")).alias("d_place"))
    m = fit_ride_time(df, PLACE_NODE, DIST, external=3, bands_m=(0, 1500, np.inf), min_trips=150)
    assert np.all(np.isfinite(m.speed_q)) and np.all(np.diff(m.speed_q, axis=2) >= 0)


def test_roundtrip_file(model, tmp_path):
    model.save(tmp_path / "rt.npz")
    back = RideTimeModel.load(tmp_path / "rt.npz")
    np.testing.assert_array_equal(back.speed_q, model.speed_q)
    rows = summary(back)
    assert rows[0]["bike_type"] == "classic" and "median_kmh" in rows[0]
