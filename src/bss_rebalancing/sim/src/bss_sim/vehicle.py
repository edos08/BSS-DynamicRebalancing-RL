"""Van energy, operating cost and emissions (issue #122, spec §9 "Energy", "Cost and emissions").

The van battery is **not a constraint** (D20): vans are assumed always charged. Energy, cost and
CO2e are **KPIs**. A run is flagged when a van uses more than its usable battery between two
depot visits, so the assumption is checked rather than trusted.

**Consumption of a drive** of ``d`` metres in ``t`` seconds (average speed ``v = d / t``) with ``L``
bikes on board, mass ``m = m_van + L x bike_kg``:

```
E = [ m g Crr d  +  1/2 rho CdA v^2 d  +  stops_per_km (d/1000) 1/2 m v^2 (1 - regen) ] / eta
    +  aux_kw t
```

rolling resistance, aerodynamic drag at the average speed, kinetic energy lost at stops (city
intersections) net of regenerative braking, all through the drivetrain efficiency, plus
auxiliaries (HVAC, electronics) for the driving time. During services the van stands with
``service_aux_kw``.

**Preset ``e_transit``** (Ford E-Transit cargo van; ``vehicle`` section of the configuration). Usable battery 68 kWh (2022–23 models;
89 kWh for the 2024+ extended-range pack). Ford's range figures give about 0.34 kWh/km on a mixed
cycle (68 kWh / 126 mi, 89 kWh / 159 mi). Body and drivetrain parameters are typical values for a
large cargo van (mass 2,900 kg, Crr 0.012, CdA 2.0 m², eta 0.85, regeneration 60 %, auxiliaries
2 kW). With them the model gives 0.31 kWh/km at 60 km/h with one stop per km, close to the Ford
figure, and about 0.24 kWh/km at 25 km/h with three stops per km (Cambridge streets).

**Costs** (USD, Cambridge MA; ``costs`` section of the configuration):

* electricity 0.22 USD/kWh: EIA, Massachusetts commercial price, October 2025 (21.98 ¢/kWh);
  energy from the grid = battery energy / charging efficiency (0.90, assumption);
* driver 22.68 USD/h: BLS OEWS May 2021, light truck drivers (53-3033), Boston-Cambridge-Nashua
  (wages only, no overhead), per truck-hour from the start of the run;
* van 0.10 USD/km: maintenance and tyres (assumption; no local source);
* battery swaps 0 USD each by default: their handling time is already paid as driver time.

**Emissions**: 0.254 kg CO2e per grid kWh, ISO New England 2024 average including net imports
(560 lb CO2/MWh).
"""

from __future__ import annotations

from bss_sim.config import CostConfig, VehicleConfig

G = 9.81
RHO_AIR = 1.2


class VanEnergy:
    def __init__(self, params: VehicleConfig):
        self.p = params

    def drive_kwh(self, dist_m: float, time_s: float, load: int = 0) -> float:
        if dist_m <= 0.0:
            return self.p.aux_kw * time_s / 3600.0
        p = self.p
        m = p.mass_kg + load * p.bike_kg
        v = dist_m / time_s if time_s > 0 else 0.0
        joules = (m * G * p.crr * dist_m + 0.5 * RHO_AIR * p.cda_m2 * v * v * dist_m
                  + p.stops_per_km * dist_m / 1000.0 * 0.5 * m * v * v * (1.0 - p.regen_eff)) / p.drivetrain_eff
        return joules / 3.6e6 + p.aux_kw * time_s / 3600.0

    def service_kwh(self, time_s: float) -> float:
        return self.p.service_aux_kw * time_s / 3600.0


class CostModel:
    def __init__(self, cfg: CostConfig):
        self.c = cfg

    def grid_kwh(self, battery_kwh: float) -> float:
        return battery_kwh / self.c.charging_efficiency

    def cost_usd(self, battery_kwh: float, hours: float, km: float, swaps: int) -> dict[str, float]:
        c = self.c
        parts = {"electricity": self.grid_kwh(battery_kwh) * c.electricity_usd_per_kwh,
                 "driver": hours * c.driver_usd_per_hour,
                 "van": km * c.van_usd_per_km,
                 "swaps": swaps * c.battery_swap_usd}
        parts["total"] = sum(parts.values())
        return parts

    def kgco2e(self, battery_kwh: float) -> float:
        return self.grid_kwh(battery_kwh) * self.c.grid_kgco2e_per_kwh
