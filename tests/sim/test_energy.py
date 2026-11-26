import numpy as np
import pytest

from bss_sim.config import EnergyConfig
from bss_sim.energy import Battery
from bss_sim.state import ELECTRIC, MECHANICAL


def test_usability_needs_trip_plus_reserve():
    b = Battery(EnergyConfig(consumption_km_per_km=1.5, reserve_km=1.0), 60.0)
    need = b.need_km(4.0)
    assert need == pytest.approx(7.0)
    assert b.usable(ELECTRIC, 7.0, need) and not b.usable(ELECTRIC, 6.9, need)
    assert b.usable(MECHANICAL, 0.0, need)
    assert Battery(EnergyConfig(dead_ebike_rideable=True), 60.0).usable(ELECTRIC, 0.0, need)


def test_consumption_never_below_zero():
    b = Battery(EnergyConfig(consumption_km_per_km=2.0), 60.0)
    assert b.after_ride(10.0, 3.0) == pytest.approx(4.0)
    assert b.after_ride(1.0, 3.0) == 0.0


def test_bands_scalar_and_vector_agree():
    b = Battery(EnergyConfig(battery_bands=(0.2, 0.5)), 50.0)  # edges at 10 km and 25 km
    assert b.n_bands == 3
    km = np.array([0.0, 9.99, 10.0, 24.9, 25.0, 50.0])
    assert b.bands(km).tolist() == [0, 0, 1, 1, 2, 2]
    assert [b.band(float(x)) for x in km] == b.bands(km).tolist()


def test_initial_draw_within_bounds():
    b = Battery(EnergyConfig(), 60.0)
    x = b.draw(np.random.default_rng(0), 10000, 0.3, 1.0)
    assert x.min() >= 18.0 and x.max() <= 60.0
    assert x.mean() == pytest.approx(39.0, abs=0.5)
