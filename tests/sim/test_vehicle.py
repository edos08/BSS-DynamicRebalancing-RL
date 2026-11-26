from dataclasses import replace

import pytest

from bss_sim.config import ConfigError, CostConfig, SimConfig, VehicleConfig
from bss_sim.vehicle import CostModel, VanEnergy


def test_consumption_anchors_documented_in_the_module():
    e = VanEnergy(VehicleConfig())
    assert e.drive_kwh(1000, 144) == pytest.approx(0.238, abs=0.002)            # 25 km/h, 3 stops/km
    assert VanEnergy(replace(VehicleConfig(), stops_per_km=1.0)).drive_kwh(1000, 60) == pytest.approx(0.306, abs=0.002)
    assert 0.30 < VanEnergy(replace(VehicleConfig(), stops_per_km=1.0)).drive_kwh(1000, 60) < 0.36  # Ford ~0.34


def test_consumption_properties():
    e = VanEnergy(VehicleConfig())
    assert e.drive_kwh(2000, 288) == pytest.approx(2 * e.drive_kwh(1000, 144))  # same speed: linear in distance
    assert e.drive_kwh(1000, 144, load=30) > e.drive_kwh(1000, 144)
    assert e.drive_kwh(1000, 60) > e.drive_kwh(1000, 100) * 0.9                 # drag and stops grow with speed
    assert e.drive_kwh(0, 3600) == pytest.approx(2.0)                           # standing: auxiliaries only
    assert e.service_kwh(3600) == pytest.approx(0.5)


def test_costs_and_emissions():
    c = CostModel(CostConfig())
    parts = c.cost_usd(battery_kwh=18.0, hours=8.0, km=80.0, swaps=40)
    assert parts["electricity"] == pytest.approx(18.0 / 0.9 * 0.22)
    assert parts["driver"] == pytest.approx(8 * 22.68) and parts["van"] == pytest.approx(8.0)
    assert parts["total"] == pytest.approx(sum(v for k, v in parts.items() if k != "total"))
    assert c.kgco2e(18.0) == pytest.approx(20.0 * 0.254)


def test_config_sections():
    cfg = SimConfig.from_dict({"vehicle": {"usable_kwh": 89}, "costs": {"driver_usd_per_hour": 30}})
    assert cfg.vehicle.usable_kwh == 89.0 and cfg.costs.driver_usd_per_hour == 30.0
    for bad in ({"vehicle": {"preset": "diesel"}}, {"vehicle": {"regen_eff": 1.5}}, {"costs": {"charging_efficiency": 0}},
                {"costs": {"van_usd_per_km": -1}}, {"vehicle": {"mass_kg": 0}}):
        with pytest.raises(ConfigError):
            SimConfig.from_dict(bad)
