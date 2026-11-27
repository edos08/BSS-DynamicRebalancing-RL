"""GridZoneEnv: the legacy grid environment on the new simulator (issue #165, spec §14).

**Zones and actions.** The city is cut into 300 m square zones (:func:`bss_sim.zones.grid_zones`);
the truck is always in one zone. The legacy actions map to truck commands:

========  ================================  ======================================================
action    command                           invalid when
========  ================================  ======================================================
STAY      ``Wait(60)``                      never
UP/DOWN/  ``GoTo(stop of the neighbouring   there is no zone in that direction
LEFT/     zone)`` (north, south, west,
RIGHT     east)
DROP      ``Drop(1)`` at the zone's stop    the truck is empty, or the bike cannot be placed
PICK_UP   ``PickUp(1)`` within ``radius_m`` no bike within reach, or the truck is full
CHARGE    ``SwapBattery(1)`` within reach   no e-bike to swap, or no charged battery left
========  ================================  ======================================================

An invalid action costs one step (30 s) and the legacy invalid-action reward.

**Semi-MDP stepping.** One environment *step* is ``step_s`` = 30 s of simulated time. An action
lasts as long as its command really takes (driving, loading), rounded up to whole steps. ``step``
returns when the action is over. ``info["steps"]`` is its duration ``tau``, ``info["elapsed_s"]``
the time, and ``info["step_rewards"]`` the reward of each of its 30 s steps. A learner discounts
them as ``R = sum_i gamma^i r_i`` and the next state by ``gamma^tau`` (issue #164). The returned
``reward`` is their plain sum, as Gymnasium expects.

**Observation** (``Dict``):

* ``graph`` ``[Z, 4]``: per zone the legacy node features: truck here, critic score, eligibility,
  bikes. The edges are the grid adjacency (``edge_index``, ``edge_attr`` = expected drive time
  between the zones' stops, minutes), which is constant.
* ``agent``: the legacy vector, i.e. scalars (bikes deployed, depot share, surplus and empty flags
  of the truck's zone, mean critic), previous action, truck zone, critical zones, missing
  neighbours. Three features are added: truck load, and time of day as sin/cos.

**Critic** (legacy formula, ``critic = -1 + exp((1 + 0.2 a)(1 - bikes / expected))``):
``expected = max(D - A, 0) + min_bikes`` and ``a = A``. ``D`` and ``A`` are the forecast departures
and arrivals of the zone over the next ``critic_window_s`` (3 h), from ``sim.forecast`` (no oracle),
refreshed every ``forecast_every_s``. Unlike the legacy cumulative peak, the window's net outflow is
used. With ``binary_critic`` (default, the legacy ``USE_BINARY_CRITIC`` of the paper runs) a zone with
a positive score is critical and scores 1, and the global critic is the number of critical zones.
Without it the raw score is used (critical above 0.05), up to e^10 per zone, which makes the
change-of-global-critic reward term reach +-10^4 per step.

**Reward** (:class:`RewardConfig`): the legacy shaped reward of the action (drop or pick-up by
criticality, charge usefulness, eligibility, stay, loops, deployment, change of global critic),
counted in the action's first step, plus **minus ``failure_weight`` per failed request** in every
30 s step of the action. Failures are now a per-step cost rather than an end-of-episode lump.

**Episodes** last ``episode_days`` from a start chosen in ``start_dates`` (one per reset, from the
reset seed). They end by time limit: ``truncated`` is true and ``terminated`` stays false.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from bss_sim.commands import Drop, GoTo, PickUp, SwapBattery, Wait
from bss_sim.config import SimConfig
from bss_sim.engine import TimeLimit
from bss_sim.simulator import Simulator
from bss_sim.state import DEPOT, ELECTRIC, PARKED
from bss_sim.zones import grid_zones

STAY, UP, DOWN, LEFT, RIGHT, DROP, PICK_UP, CHARGE = range(8)
ACTION_NAMES = ("STAY", "UP", "DOWN", "LEFT", "RIGHT", "DROP_BIKE", "PICK_UP_BIKE", "CHARGE_BIKE")
DIRECTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}
SELF_LOOPS = {(UP, DOWN), (DOWN, UP), (LEFT, RIGHT), (RIGHT, LEFT), (STAY, STAY), (PICK_UP, DROP), (DROP, PICK_UP)}
CRITIC_THRESHOLD = 0.05


@dataclass(frozen=True)
class RewardConfig:
    """Legacy ``RewardComponents`` (thesis values) plus the per-step failure cost."""

    base_cost: float = -0.1
    invalid_action: float = -1.0
    loop_penalty: float = -0.6
    drop_base: float = 0.01
    drop_rebalanced_critical: float = 2.0
    drop_in_critical: float = 1.0
    drop_in_surplus: float = -0.5
    pickup_from_critical: float = -0.5
    pickup_unbalanced_cell: float = -2.0
    pickup_from_surplus: float = 0.2
    charge_useless_critical: float = -0.1
    charge_useless_normal: float = -0.3
    charge_useful_critical: float = 0.5
    charge_useful_normal: float = 0.3
    charge_low_battery_threshold: float = 0.8
    eligibility_high: float = 0.7
    eligibility_low: float = 0.2
    eligibility_revisit_penalty: float = -0.2
    eligibility_exploration_bonus: float = 0.3
    eligibility_empty_truck_penalty: float = -0.05
    stay_base: float = -0.1
    stay_in_critical: float = -1.0
    stay_no_critic: float = 0.0
    deploy_weight: float = 0.02
    depot_weight: float = 0.02
    global_critic_weight: float = 0.5
    failure_weight: float = 0.02        # per failed request, in the 30 s step it happens


@dataclass
class _Zone:
    ix: int
    iy: int
    neighbours: dict[int, int] = field(default_factory=dict)   # action -> zone


class GridZoneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bundle, sim_config: dict | SimConfig | None = None,
                 start_dates: list[str] | tuple[str, ...] = ("2024-09-16",), start_hour: float = 0.0,
                 episode_days: float = 7.0, cell_m: float = 300.0, step_s: float = 30.0,
                 radius_m: float = 250.0, critic_window_s: float = 3 * 3600.0, forecast_every_s: float = 900.0,
                 min_bikes: int = 1, reward: RewardConfig | None = None, eligibility_decay: float = 0.9968,
                 border_eligibility_decay: float = 0.99, slot_s: float = 3 * 3600.0, binary_critic: bool = True):
        cfg = sim_config if isinstance(sim_config, SimConfig) else SimConfig.from_dict(sim_config or {})
        if cfg.trucks.count != 1:
            cfg = cfg.replace(**{"trucks.count": 1})
        self.sim = Simulator.from_scenario(bundle, cfg)
        sc = self.sim.scenario
        self.zones = grid_zones(sc, cell_m)
        self.sim.add_partition(self.zones.name, self.zones.node_to_zone)
        self.start_dates = tuple(start_dates)
        self.start_hour = start_hour
        self.episode_s = episode_days * 86400.0
        self.step_s = step_s
        self.radius_m = radius_m
        self.critic_window_s = critic_window_s
        self.forecast_every_s = forecast_every_s
        self.min_bikes = min_bikes
        self.reward_cfg = reward or RewardConfig()
        self.decay = eligibility_decay
        self.border_decay = border_eligibility_decay
        self.slot_s = slot_s
        self.binary_critic = binary_critic
        self._build_grid(cell_m)
        Z = self.zones.n
        self.n_zones = Z
        self.agent_len = 5 + 8 + Z + Z + 4 + 3
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Dict({
            "graph": gym.spaces.Box(-np.inf, np.inf, shape=(Z, 4), dtype=np.float32),
            "agent": gym.spaces.Box(-np.inf, np.inf, shape=(self.agent_len,), dtype=np.float32),
        })
        self._weights = sc.departure_weights()
        self.fleet_total = None

    # ------------------------------------------------------------------------------ grid
    def _build_grid(self, cell_m: float) -> None:
        sc, z = self.sim.scenario, self.zones
        x0, y0 = sc.bike_x.min(), sc.bike_y.min()
        cells = {}
        for k in range(z.n):
            node = z.nodes(k)[0]
            ix = int(np.floor((sc.bike_x[node] - x0) / cell_m))
            iy = int(np.floor((sc.bike_y[node] - y0) / cell_m))
            cells[(ix, iy)] = k
        self._grid = [None] * z.n
        for (ix, iy), k in cells.items():
            zone = _Zone(ix, iy)
            for a, (dx, dy) in DIRECTIONS.items():
                nb = cells.get((ix + dx, iy + dy))
                if nb is not None:
                    zone.neighbours[a] = nb
            self._grid[k] = zone
        src, dst = [], []
        for k, zone in enumerate(self._grid):
            for nb in zone.neighbours.values():
                src.append(k)
                dst.append(nb)
        self.edge_index = np.array([src, dst], dtype=np.int64)
        stops = z.stop
        a = sc.bike_to_drive[stops[src]]
        b = sc.bike_to_drive[stops[dst]]
        self.edge_attr = (np.asarray(sc.drive_time)[a, b] / 60.0).astype(np.float32).reshape(-1, 1)
        self.zone_of_node = z.node_to_zone

    def borders(self, zone: int) -> np.ndarray:
        """1 for each direction (UP, DOWN, LEFT, RIGHT) without a neighbouring zone (legacy flags)."""
        nb = self._grid[zone].neighbours
        return np.array([0.0 if a in nb else 1.0 for a in (UP, DOWN, LEFT, RIGHT)], dtype=np.float32)

    # ----------------------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        start = options.get("start")
        if start is None:
            day = self.start_dates[int(rng.integers(len(self.start_dates)))]
            start = dt.datetime.fromisoformat(day) + dt.timedelta(hours=self.start_hour)
        sim_seed = int(options.get("sim_seed", rng.integers(2**31)))
        self.sim.reset(seed=sim_seed, start=start)
        self.sim.run(TimeLimit(0))                     # drain the start notifications
        self.t_end = self.sim.now + self.episode_s
        self.fleet_total = self.sim.fleet.n - self.sim.config.external.reservoir
        Z = self.n_zones
        self.eligibility = np.zeros(Z)
        self.prev_action = None
        self.last_move = None
        self.last_border = 0.0
        self._fc_t = -np.inf
        self._update_critic(force=True)
        self.truck_zone = int(self.zone_of_node[self.sim.trucks[0].stop])
        if self.truck_zone < 0:
            self.truck_zone = 0
        self.eligibility[self.truck_zone] = 1.0
        self._failed0 = self.sim.kpis_.failures.copy()
        self.slot = int(self.sim.now // self.slot_s)
        return self._obs(), self._info(steps=0, elapsed=0.0, step_rewards=[], failures=[], invalid=False)

    # ----------------------------------------------------------------------------- critic
    def _update_critic(self, force: bool = False) -> None:
        sim = self.sim
        if force or sim.now - self._fc_t >= self.forecast_every_s:
            fc = sim.forecast(self.critic_window_s, self.zones.name)
            self._expected = np.maximum(fc.departures - fc.arrivals, 0.0) + self.min_bikes
            self._after = fc.arrivals
            self._fc_t = sim.now
        bikes = sim.partition(self.zones.name).bikes.astype(np.float64)
        self.bikes = bikes
        with np.errstate(over="ignore"):
            expo = np.minimum((1.0 + 0.2 * self._after) * (1.0 - bikes / self._expected), 10.0)
            raw = np.where(self._expected > 0, -1.0 + np.exp(expo), -bikes)
        self.raw_critic = raw
        if self.binary_critic:
            # legacy USE_BINARY_CRITIC (the paper setting): a zone with a positive score is critical and
            # scores 1; the global critic counts critical zones. Without it the score is exp(...) - 1, up
            # to e^10 per zone, and the -0.5 * delta(global critic) reward term reaches +-10^4 per step.
            self.critic = np.where(raw > 0.0, 1.0, raw)
            self.critical = raw > 0.0
            self.global_critic = float(self.critical.sum())
        else:
            self.critic = raw
            self.critical = raw > CRITIC_THRESHOLD
            self.global_critic = float(np.maximum(raw, 0.0).sum())
        t = 0.67   # legacy surplus threshold (on the raw score, as in Cell.set_surplus_bikes)
        surplus = bikes - np.floor(bikes * (1 + raw) * (1 - t) / (1 + raw * (1 - t) / t))
        self.surplus = np.where(raw > -t, 0.0, np.maximum(surplus, 0.0))

    # ----------------------------------------------------------------------------- step
    def _command(self, action: int):
        tr = self.sim.trucks[0]
        tc = self.sim.config.trucks
        z = self.truck_zone
        if action == STAY:
            return [Wait(60.0)]
        if action in DIRECTIONS:
            nb = self._grid[z].neighbours.get(action)
            return None if nb is None else [GoTo(int(self.zones.stop[nb]))]
        if action == DROP:
            return None if not tr.load else [Drop(1)]
        if action == PICK_UP:
            return None if len(tr.load) >= tc.capacity else [PickUp(1, policy="nearest", radius_m=self.radius_m)]
        if action == CHARGE:
            return None if tr.stock <= 0 else [SwapBattery(1, radius_m=self.radius_m)]
        raise ValueError(f"unknown action {action}")

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        sim = self.sim
        rc = self.reward_cfg
        # state before the action (for the shaped reward)
        z0 = self.truck_zone
        old_crit, old_global = float(self.critic[z0]), self.global_critic
        was_critical, was_surplus = old_crit > 0.0, self.surplus[z0] > 0
        old_elig = float(self.eligibility[z0])
        charge_amount = self._charge_amount() if action == CHARGE else 0.0

        cmds = self._command(action)
        t0 = sim.now
        invalid = cmds is None
        if not invalid:
            sim.submit(0, cmds)
        failures = []
        results = []
        # run in 30 s steps until the action is over (at least one step)
        while True:
            before = sum(sim.kpis_.failures[:5])
            notes = sim.run(TimeLimit(self.step_s))
            failures.append(sum(sim.kpis_.failures[:5]) - before)
            results += [n.data["result"] for n in notes if n.kind == "command_done"]
            if not sim.trucks[0].busy or sim.now >= self.t_end:
                break
        if not invalid and results and results[-1].executed == 0 and results[-1].requested > 0:
            invalid = True                              # nothing could be done (no bike, no dock...)
        steps = len(failures)
        elapsed = sim.now - t0

        # update zone state
        self.truck_zone = int(self.zone_of_node[sim.trucks[0].stop])
        if self.truck_zone < 0:
            self.truck_zone = z0
        self._decay(steps)
        self._update_critic()
        z1 = self.truck_zone

        shaped = self._shaped(action, invalid, z0, z1, was_critical, was_surplus, old_elig, old_global,
                              charge_amount)
        self.eligibility[z1] = 1.0
        step_rewards = [-rc.failure_weight * f for f in failures]
        step_rewards[0] += shaped
        self.prev_action = action
        if not invalid:
            self.last_move = action
        self.last_border = float(self.borders(z1).sum())

        truncated = sim.now >= self.t_end
        slot = int(sim.now // self.slot_s)
        slot_end = slot != self.slot
        self.slot = slot
        info = self._info(steps, elapsed, step_rewards, failures, invalid)
        info["slot_end"] = slot_end
        info["action_name"] = ACTION_NAMES[action]
        return self._obs(), float(sum(step_rewards)), False, truncated, info

    def _decay(self, steps: int) -> None:
        border = np.array([len(self._grid[k].neighbours) < 4 for k in range(self.n_zones)])
        self.eligibility *= np.where(border, self.border_decay, self.decay) ** steps
        self.eligibility[self.eligibility < 0.001] = 0.0

    def _charge_amount(self) -> float:
        """Share of the range a swap would give the emptiest e-bike in reach (legacy ``last_charge``)."""
        sim = self.sim
        near = sim.nodes_within(sim.trucks[0].stop, self.radius_m)
        f = sim.fleet
        rng = sim.config.fleet.battery_range_km
        best = None
        for v in near:
            for b in f.bikes_at(v):
                if f.kind_l[b] == ELECTRIC:
                    best = f._batt[b] if best is None else min(best, f._batt[b])
        return 0.0 if best is None else (rng - best) / rng

    def _shaped(self, action, invalid, z0, z1, was_critical, was_surplus, old_elig, old_global, charge_amount) -> float:
        rc = self.reward_cfg
        if invalid:
            return rc.invalid_action
        loop = (action, self.last_move) in SELF_LOOPS if self.last_move is not None else False
        is_critical = bool(self.critical[z1])
        r = rc.base_cost + (rc.loop_penalty if loop else 0.0)
        if action == DROP:
            if was_critical and not is_critical:
                r += rc.drop_rebalanced_critical
                self.eligibility[self.eligibility > 0.3] -= 0.2
            elif was_critical:
                r += rc.drop_in_critical
            elif was_surplus:
                r += rc.drop_in_surplus
            else:
                r += rc.drop_base
        elif action == PICK_UP:
            if was_critical:
                r += rc.pickup_from_critical
            elif is_critical:
                r += rc.pickup_unbalanced_cell
            elif was_surplus and not loop:
                r += rc.pickup_from_surplus
        elif action == CHARGE:
            useful = charge_amount >= rc.charge_low_battery_threshold
            r += (rc.charge_useful_critical if was_critical else rc.charge_useful_normal) if useful else \
                (rc.charge_useless_critical if was_critical else rc.charge_useless_normal)
        elif action in DIRECTIONS:
            if old_elig > rc.eligibility_high and not was_critical:
                r += rc.eligibility_revisit_penalty + 0.1 * self.last_border
            elif old_elig < rc.eligibility_low:
                r += rc.eligibility_exploration_bonus
            if not self.sim.trucks[0].load:
                r += rc.eligibility_empty_truck_penalty
        elif action == STAY:
            if was_critical:
                r += rc.stay_in_critical
            elif old_global <= 0.0:
                r += rc.stay_no_critic
                if loop:
                    r -= rc.loop_penalty            # legacy: no loop penalty when nothing is critical
            else:
                r += rc.stay_base
        f = self.sim.fleet
        deployed = f.status_count[PARKED] / self.fleet_total
        depot = f.status_count[DEPOT] / self.fleet_total
        r += rc.deploy_weight * deployed - rc.depot_weight * depot
        r -= rc.global_critic_weight * (self.global_critic - old_global)
        return float(r)

    # ----------------------------------------------------------------------------- outputs
    def _obs(self) -> dict:
        Z = self.n_zones
        z = self.truck_zone
        graph = np.zeros((Z, 4), dtype=np.float32)
        graph[z, 0] = 1.0
        graph[:, 1] = self.critic
        graph[:, 2] = self.eligibility
        graph[:, 3] = self.bikes
        f = self.sim.fleet
        tr = self.sim.trucks[0]
        scal = [
            (f.status_count[PARKED] + len(tr.load)) / self.fleet_total,
            f.status_count[DEPOT] / self.fleet_total,
            1.0 if self.surplus[z] > 0 else 0.0,
            1.0 if self.bikes[z] == 0 else 0.0,
            self.global_critic / Z,
        ]
        prev = np.zeros(8)
        if self.prev_action is not None:
            prev[self.prev_action] = 1.0
        here = np.zeros(Z)
        here[z] = 1.0
        tod = 2 * math.pi * (self.sim.now % 86400.0) / 86400.0
        extra = [len(tr.load) / self.sim.config.trucks.capacity, math.sin(tod), math.cos(tod)]
        agent = np.concatenate([scal, prev, here, self.critical.astype(float), self.borders(z), extra])
        return {"graph": graph, "agent": agent.astype(np.float32)}

    def _info(self, steps, elapsed, step_rewards, failures, invalid) -> dict:
        k = self.sim.kpis_
        return {"steps": steps, "elapsed_s": elapsed, "step_rewards": step_rewards, "failures": failures,
                "invalid": invalid, "time": self.sim.now, "truck_zone": self.truck_zone,
                "truck_load": len(self.sim.trucks[0].load), "global_critic": self.global_critic,
                "requests": k.requests, "failed": int(sum(k.failures[:5])),
                "city_service_level": k.city_served / max(k.city_requests, 1)}

    def kpis(self) -> dict:
        return self.sim.kpis()
