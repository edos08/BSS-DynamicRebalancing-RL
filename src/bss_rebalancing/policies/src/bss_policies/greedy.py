"""GreedyDeficit: event-driven rebalancing from the forecasts (issue #153, spec §14).

Whenever a truck becomes idle, the controller looks at the zones again (fresh observation, mean
forecast over the next ``horizon_h`` hours), computes the surplus or deficit of every zone against
its target, removes what the other trucks are already on their way to do, and sends the truck to
the single visit that moves the most bikes per second of driving and service (the first step of
:func:`~bss_policies.planning.plan_route`). With nothing worth doing, the truck waits ``idle_wait_s``
and looks again. A zone where a pick-up or drop achieved nothing (e.g. the bikes counted in the zone
are out of reach of the stop, or users took them) is left alone for ``cooldown_s``. Unlike :class:`~bss_policies.static_tsp.StaticTSP` there is no schedule: decisions
follow the trucks and the latest information.
"""

from __future__ import annotations

import numpy as np

from bss_policies.planning import plan_route
from bss_policies.zonebase import ZoneController
from bss_sim.commands import Wait


class GreedyDeficit(ZoneController):
    name = "greedy_deficit"

    def __init__(self, horizon_h: float = 1.0, idle_wait_s: float = 600.0, min_move: int = 2,
                 cooldown_s: float = 900.0, **kw):
        super().__init__(horizon_h=horizon_h, **kw)
        self.idle_wait_s = idle_wait_s
        self.min_move = min_move
        self.cooldown_s = cooldown_s

    def describe(self) -> dict:
        return {**super().describe(), "idle_wait_s": self.idle_wait_s, "min_move": self.min_move,
                "cooldown_s": self.cooldown_s}

    def reset(self, sim) -> None:
        self.setup_zones(sim)
        self.claims: dict[int, tuple[int, int]] = {}    # truck -> (zone, qty) it is heading to do
        self.cooldown: dict[int, float] = {}            # zone -> until when it is left alone
        self.visits = 0
        self.setup_shifts(sim)

    def act(self, sim, notes) -> None:
        for n in notes:                                  # a service that achieved nothing: cool the zone down
            if n.kind == "command_done":
                r = n.data["result"]
                if type(r.command).__name__ in ("PickUp", "Drop") and r.executed == 0 and r.requested > 0:
                    claim = self.claims.get(r.truck)
                    if claim is not None:
                        self.cooldown[claim[0]] = sim.now + self.cooldown_s
        ended, started = self.shift_events(sim, notes)
        if ended:
            self.claims.clear()
        if not self.on_shift:
            return
        idle = {n.data["truck"] for n in notes if n.kind == "truck_idle"}
        if started:                                      # the vans waited at the depot: all available
            idle |= {tv.id for tv in sim.observe().trucks if not tv.busy}
        idle = sorted(idle)
        if not idle:
            return
        obs = sim.observe()
        surplus = self.surplus(sim, obs)
        for t in idle:
            self.claims.pop(t, None)
        for z, q in self.claims.values():               # what the other trucks will already do
            surplus[z] -= q
        stop = self.stops(obs, surplus)
        tc = sim.config.trucks
        for t in idle:
            tv = obs.trucks[t]
            s = np.where(np.abs(surplus) >= self.min_move, surplus, 0)
            for z, until in self.cooldown.items():
                if until > sim.now:
                    s[z] = 0
            T = sim.expected_drive_times(np.append(stop, tv.stop), stop)
            route = plan_route(s, T, tc.capacity, tv.load, tc.pickup_per_bike_s, tc.drop_per_bike_s,
                               tc.stop_overhead_s, two_opt=False)
            if not route:
                sim.submit(t, [Wait(self.idle_wait_s)])
                continue
            v = route[0]
            sim.submit(t, self.commands(sim, obs, v, stop[v.zone], tv.stock))
            self.claims[t] = (v.zone, v.qty)
            surplus[v.zone] -= v.qty
            self.visits += 1
