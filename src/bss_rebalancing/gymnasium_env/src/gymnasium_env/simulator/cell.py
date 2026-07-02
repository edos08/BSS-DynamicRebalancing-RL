from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from gymnasium_env.simulator.station import Station


# ── Threshold constants ────────────────────────────────────────────────────────
_CRITIC_THRESHOLD = 0.05
_ELIGIBILITY_MIN = 0.001
_DEFAULT_SURPLUS_THRESHOLD = 0.67


class Cell:
    """
    Represents a geographic cell in the bike-sharing system grid.

    A cell aggregates a group of stations and tracks both structural attributes
    (geometry, adjacency, nodes) and dynamic operational metrics (bikes, demand,
    critic score, eligibility). All metric mutations go through this class to
    ensure consistency.
    """

    def __init__(self, cell_id: int, boundary: Polygon, cell_size: int) -> None:
        # Structural — set once at build time, never change during training
        self._id = cell_id
        self._boundary = boundary
        self._nodes: list[int] = []
        self._center_node: int = 0
        self._cell_size = cell_size
        self._diagonal = int(math.sqrt(2) * cell_size)
        self._adjacent_cells: dict[str, int | None] = {
            'up': None, 'down': None, 'left': None, 'right': None
        }

        # Dynamic state
        self._is_critical = False
        self._metrics: dict[str, float | int] = {
            # Demand / flow (set once per timeslot)
            'demand_rate': 0.0,
            'arrival_rate': 0.0,
            # Bike counts (recomputed each step)
            'total_bikes': 0,
            'surplus_bikes': 0,
            'dead_bikes': 0.0,
            # Operational counters (accumulate over episode)
            'visits': 0,
            'operations': 0,
            'pick_ups': 0,
            'drops': 0,
            'total_departures': 0,
            'total_rebalanced': 0,
            'failures': 0,
            'total_demand': 0,
            # Derived quality metrics (recomputed each step)
            'failure_rate': 0.0,
            # Critic score + history
            'critic_score': 0.0,
            'old_critic_score': 0.0,
            # Eligibility traces
            'eligibility_score': 0.0,
            'old_eligibility_score': 0.0,
            # Truck presence flag
            'truck_cell': 0,
        }

    def __str__(self) -> str:
        return (
            f"Cell {self._id} | "
            f"Bikes: {self._metrics['total_bikes']} | "
            f"Critic: {self._metrics['critic_score']:.3f} | "
            f"Ops: {self._metrics['operations']}"
        )

    # ── One-time structural setup ───────────────────────────────────────────────

    def set_center_node(self, graph) -> None:
        """Find and assign the node closest to the cell centroid."""
        center_coords = self._boundary.centroid.coords[0]
        nearest_node = ox.distance.nearest_nodes(graph, center_coords[0], center_coords[1])
        if nearest_node not in self._nodes:
            raise ValueError("Center node not found in cell nodes")
        self._center_node = nearest_node

    # ── Episode reset ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero all metrics and flags. Called at the start of every episode."""
        for key in self._metrics:
            self._metrics[key] = 0 if isinstance(self._metrics[key], int) else 0.0
        self._is_critical = False

    # ── Per-step centralized metric update ─────────────────────────────────────

    def update_metrics(
        self,
        stations: dict[int, Station],
        expected: int,
        aft_arrivals: int,
    ) -> None:
        """
        Recompute all derived metrics from current station state.
        Called once per step via update_cells_metrics() in the environment.
        """
        # ── Bike count ────────────────────────────────────────────────────────
        total_bikes = 0
        dead_bikes = 0
        for node in self._nodes:
            bikes = stations[node].get_bikes()
            total_bikes += len(bikes)
            for bike in bikes.values():
                if bike.get_battery()/bike.get_max_battery() <= 0.2:
                    dead_bikes += 1
        self.set_total_bikes(total_bikes)
        self._metrics['dead_bikes'] = dead_bikes

        # ── Failure rate ──────────────────────────────────────────────────────
        self._metrics['failure_rate'] = (
            self.get_failures() / self.get_total_demand() if self.get_total_demand() > 0 else 0.0
        )

        # ── Critic score & surplus ────────────────────────────────────────────
        if expected > 0:
            critic_score = -1 + np.exp((1 + 0.2 * aft_arrivals) * (1 - total_bikes / expected))
        else:
            critic_score = -float(total_bikes)
        self.set_critic_score(critic_score)
        self.set_surplus_bikes()

    # ── Eligibility traces ──────────────────────────────────────────────────────

    def set_eligibility_score(self, score: float) -> None:
        """Set eligibility to a new value, snapshotting the previous one."""
        self._metrics['old_eligibility_score'] = self._metrics['eligibility_score']
        self._metrics['eligibility_score'] = score

    def update_eligibility_score(self, decay: float) -> None:
        """Apply multiplicative decay to the eligibility score."""
        self._metrics['old_eligibility_score'] = self._metrics['eligibility_score']
        current = self._metrics['eligibility_score']
        self._metrics['eligibility_score'] = 0.0 if current < _ELIGIBILITY_MIN else current * decay

    # ── Critic score & surplus ──────────────────────────────────────────────────

    def set_critic_score(self, critic_score: float) -> None:
        """Set critic score, update is_critical flag, and zero surplus if critical."""
        self._metrics['old_critic_score'] = self._metrics['critic_score']
        self._metrics['critic_score'] = critic_score
        if critic_score > _CRITIC_THRESHOLD:
            self._is_critical = True
            self._metrics['surplus_bikes'] = 0
        else:
            self._is_critical = False

    def set_surplus_bikes(self, surplus_threshold: float = _DEFAULT_SURPLUS_THRESHOLD) -> None:
        """Compute surplus bikes based on how far below the threshold the critic score is."""
        if not 0.0 < surplus_threshold < 1.0:
            raise ValueError("surplus_threshold must be between 0 and 1.")
        score = self._metrics['critic_score']
        total = self._metrics['total_bikes']
        if score > -surplus_threshold:
            self._metrics['surplus_bikes'] = 0
        else:
            t = surplus_threshold
            self._metrics['surplus_bikes'] = total - math.floor(
                total * (1 + score) * (1 - t) / (1 + score * (1 - t) / t)
            )

    def set_is_critical(self, is_critical: bool) -> None:
        """Directly set the is_critical flag (use with caution, may cause inconsistency)."""
        self._is_critical = is_critical

    # ── Accumulator increments ──────────────────────────────────────────────────

    def add_departure(self, d: int = 1) -> None:
        self._metrics['total_departures'] += d
        self._metrics['total_demand'] += d

    def add_failure(self, f: int = 1) -> None:
        self._metrics['failures'] += f
        self._metrics['total_demand'] += f

    def update_rebalanced_times(self)   -> None: self._metrics['total_rebalanced'] += 1

    # ── Direct setters ──────────────────────────────────────────────────────────

    def set_metric(self, metric_name: str, value: float | int) -> None:
        """Generic setter — use only for metrics without side-effect logic."""
        self._metrics[metric_name] = value

    def set_total_bikes(self, total_bikes: int) -> None: self._metrics['total_bikes']  = total_bikes
    def set_demand_rate(self, rate: float)      -> None: self._metrics['demand_rate']  = rate
    def set_arrival_rate(self, rate: float)     -> None: self._metrics['arrival_rate'] = rate
    def set_visits(self, visits: int)           -> None: self._metrics['visits']       = visits
    def set_ops(self, ops: int)                 -> None: self._metrics['operations']   = ops
    def set_pick_ups(self, pick_ups: int)       -> None: self._metrics['pick_ups']     = pick_ups
    def set_drops(self, drops: int)             -> None: self._metrics['drops']        = drops

    # ── Getters ─────────────────────────────────────────────────────────────────

    # Structural
    def get_id(self)             -> int:       return self._id
    def get_boundary(self)       -> Polygon:   return self._boundary
    def get_nodes(self)          -> list[int]: return self._nodes
    def get_center_node(self)    -> int:       return self._center_node
    def get_diagonal(self)       -> int:       return self._diagonal
    def get_adjacent_cells(self) -> dict:      return self._adjacent_cells

    # Bike counts
    def get_total_bikes(self)    -> int:       return self._metrics['total_bikes']
    def get_surplus_bikes(self)  -> int:       return self._metrics['surplus_bikes']
    def get_dead_bikes(self)     -> int:       return self._metrics['dead_bikes']

    # Demand
    def get_demand_rate(self)    -> float:     return self._metrics['demand_rate']
    def get_arrival_rate(self)   -> float:     return self._metrics['arrival_rate']

    # Operational counters
    def get_visits(self)           -> int:   return self._metrics['visits']
    def get_ops(self)              -> int:   return self._metrics['operations']
    def get_pick_ups(self)         -> int:   return self._metrics['pick_ups']
    def get_drops(self)            -> int:   return self._metrics['drops']
    def get_total_demand(self)     -> int:   return self._metrics['total_demand']
    def get_total_departures(self) -> int:   return self._metrics['total_departures']
    def get_total_rebalanced(self) -> int:   return self._metrics['total_rebalanced']
    def get_failures(self)         -> int:   return self._metrics['failures']

    # Derived quality
    def get_failure_rate(self)     -> float: return self._metrics['failure_rate']

    # Critic
    def get_critic_score(self)     -> float: return self._metrics['critic_score']
    def get_old_critic_score(self) -> float: return self._metrics['old_critic_score']

    # Eligibility
    def get_eligibility_score(self)     -> float: return self._metrics['eligibility_score']
    def get_old_eligibility_score(self) -> float: return self._metrics['old_eligibility_score']

    # All metrics (for GNN feature extraction)
    def get_metric(self, metric_name: str) -> float | int | None:
        return self._metrics.get(metric_name, None)

    def get_all_metrics(self) -> dict:
        return self._metrics.copy()

    # ── State queries ────────────────────────────────────────────────────────────

    def is_critical(self) -> bool:
        return self._is_critical

    def has_all_neighbors(self) -> bool:
        return all(adj is not None for adj in self._adjacent_cells.values())
