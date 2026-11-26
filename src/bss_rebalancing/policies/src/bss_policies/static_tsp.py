"""StaticTSP: the thesis baseline, driven for real (issue #152, spec §14).

At scheduled hours (every ``every_h`` hours from ``first_h``) the controller reads the bikes per zone
and the expected departures and arrivals over the next ``every_h`` hours, computes target bikes per
zone, and plans, truck after truck, a capacitated pick-up and delivery route on expected drive times
within the time until the next schedule (:mod:`bss_policies.planning`). The route is submitted as
commands (:mod:`bss_policies.zonebase`).

Unlike the thesis environment, nothing is teleported: driving, loading and unloading take their
real time, bikes keep moving meanwhile, and the next schedule replaces whatever is left.
"""

from __future__ import annotations

import numpy as np

from bss_policies.planning import plan_route
from bss_policies.zonebase import ZoneController


class StaticTSP(ZoneController):
    name = "static_tsp"

    def __init__(self, every_h: float = 3.0, first_h: float = 6.0, horizon_h: float | None = None, **kw):
        if every_h <= 0:
            raise ValueError("every_h must be > 0")
        super().__init__(horizon_h=horizon_h or every_h, **kw)
        self.every_s = every_h * 3600.0
        self.first_s = first_h * 3600.0

    def describe(self) -> dict:
        return {**super().describe(), "every_h": self.every_s / 3600, "first_h": self.first_s / 3600}

    def reset(self, sim) -> None:
        self.setup_zones(sim)
        self.setup_shifts(sim)
        self.plans = 0
        day_s = sim.now % 86400.0
        k = np.ceil((day_s - self.first_s) / self.every_s)
        sim.wakeup(at=sim.now - day_s + self.first_s + max(k, 0) * self.every_s, tag="rebalance")

    def act(self, sim, notes) -> None:
        self.shift_events(sim, notes)
        if any(n.kind == "wakeup" and n.data.get("tag") == "rebalance" for n in notes):
            if self.on_shift:
                self.rebalance(sim)
            sim.wakeup(after=self.every_s, tag="rebalance")

    def rebalance(self, sim) -> None:
        obs = sim.observe()
        surplus = self.surplus(sim, obs)
        stop = self.stops(obs, surplus)
        tc = sim.config.trucks
        for tv in obs.trucks:
            if not surplus.any():
                break
            T = sim.expected_drive_times(np.append(stop, tv.stop), stop)
            budget = self.every_s
            if self.shift is not None:      # finish before the end of the shift
                budget = min(budget, self.shift[1] * 3600.0 - sim.now % 86400.0)
            route = plan_route(surplus, T, tc.capacity, tv.load, tc.pickup_per_bike_s, tc.drop_per_bike_s,
                               tc.stop_overhead_s, time_budget_s=budget)
            cmds = []
            for v in route:
                cmds += self.commands(sim, obs, v, stop[v.zone], tv.stock)
                surplus[v.zone] -= v.qty
            sim.submit(tv.id, cmds, replace=True)
        self.plans += 1
