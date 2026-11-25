"""
Static Bike-Sharing System Environment for Reinforcement Learning.

This module implements a Gymnasium environment for simulating static bike
rebalancing in a bike-sharing system. The environment simulates bike trips
and periodic system-wide rebalancing operations.

Author: Edoardo Scarpel
"""

import heapq
import logging
import multiprocessing
import os

import gymnasium as gym
import numpy as np
import osmnx as ox
import polars as pl

from collections import deque
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.bike_simulator import build_events, event_handler
from gymnasium_env.simulator.env_logger import EnvLogger
from gymnasium_env.simulator.event import Event
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import TripSample
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.truck_simulator import tsp_rebalancing
from gymnasium_env.simulator.utils import (
    DAYS_TO_NUM,
    DEFAULT_PATHS,
    NUM_TO_DAYS,
    Actions,
    compute_buffer_logic,
    convert_seconds_to_hours_minutes_day,
    episode_worker,
    initialize_bikes,
    initialize_graph,
    initialize_stations,
    load_preprocessed_data,
    truncated_gaussian,
)

# =============================================================================
# Environment Constants
# =============================================================================


class EnvDefaults:
    """Default configuration values for the environment."""

    # Bike fleet parameters
    MAX_BIKES = 1000
    MIN_BIKES_PER_CELL = 5
    BASE_REPOSITIONING = True
    NET_FLOW_BASED_REPOSITIONING = True

    # Truck parameters
    MAX_TRUCK_LOAD = 30
    INITIAL_TRUCK_BIKES = 0

    # Time parameters
    TIMESLOT_DURATION_HOURS = 3
    TIMESLOT_DURATION_SECONDS = TIMESLOT_DURATION_HOURS * 3600
    STEP_DURATION_SECONDS = 30
    TIMESLOTS_PER_DAY = 8

    # RL parameters
    DISCOUNT_FACTOR = 0.99
    ELIGIBILITY_DECAY = 0.9968
    BORDER_ELIGIBILITY_DECAY = 0.99

    # Default starting conditions
    DEFAULT_DAY = "monday"
    DEFAULT_TIMESLOT = 0
    DEFAULT_TOTAL_TIMESLOTS = 56
    DEFAULT_DEPOT_ID = 1
    DEFAULT_NUM_REBALANCING_EVENTS = 2
    DEFAULT_STARTING_REBALANCING_EVENT = 1  # hour at which first rebalancing event occurs (e.g. 1 = 1am, 6 = 6am, etc.)

    # Simulation parameters
    PRECOMPUTED_EPISODE_TIMESLOTS = 56  # 7 days × 8 slots
    FINGERPRINT_FILE = "event_buffer_fingerprint.json"


@dataclass
class Depot:
    id: int | None = None
    bikes: dict[int, Bike] = field(default_factory=dict)


# =============================================================================
# Main Environment Class
# =============================================================================


class StaticEnv(gym.Env):
    """
    A Gymnasium environment for bike-sharing system static rebalancing.

    This environment simulates a bike-sharing network where periodic
    system-wide rebalancing occurs at fixed intervals. Unlike the dynamic
    environment, there is no continuous truck control - instead, the system
    is rebalanced using a TSP-based approach at scheduled times.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        data_path: str,
        results_path: str,
        seed: int = 42,
        logging_enabled: bool = False,
    ):
        """
        Initialize the static bike-sharing environment.

        Args:
            data_path: Path to directory containing network data, PMF matrices, etc.
        """
        super().__init__()

        # Store path
        self._data_path = data_path
        self._results_path = results_path

        # Env logger: lazy init, like FullyDynamicEnv
        log_dir = (
            os.path.join(results_path, "logs") if results_path is not None else None
        )
        self._env_logger = EnvLogger(name="static-env")
        self._env_logger.init(
            log_dir=log_dir,
            filename="env.log",
            level=logging.INFO,
            enabled=logging_enabled,
        )

        # -------------------------------------------------------------------------
        # Load and initialize network components
        # -------------------------------------------------------------------------

        # Load the OSMnx graph
        self._graph = initialize_graph(
            os.path.join(self._data_path, DEFAULT_PATHS.graph_file)
        )

        # Compute bounding box for coordinate normalization
        nodes_gdf = ox.graph_to_gdfs(self._graph, edges=False)
        self._min_lat, self._max_lat = nodes_gdf["y"].min(), nodes_gdf["y"].max()
        self._min_lon, self._max_lon = nodes_gdf["x"].min(), nodes_gdf["x"].max()
        self._nodes_dict = {
            nid: (row["y"], row["x"]) for nid, row in nodes_gdf.iterrows()
        }

        # -------------------------------------------------------------------------
        # Load precomputed data
        # -------------------------------------------------------------------------

        (
            self._distance_matrix,
            self._velocity_matrix,
            self._consumption_matrix,
            self._nearby_nodes_dict,
            self._global_rate_dict,
            self._cells,
        ) = load_preprocessed_data(
            DEFAULT_PATHS,
            self._data_path,
        )

        self._distance_lookup = {
            row["node_id"]: row for row in self._distance_matrix.iter_rows(named=True)
        }

        self._velocity_lookup = {
            row["hour"]: row for row in self._velocity_matrix.iter_rows(named=True)
        }

        self._consumption_lookup = {
            row["hour"]: row for row in self._consumption_matrix.iter_rows(named=True)
        }

        self._sorted_cell_ids = sorted(self._cells.keys())

        # -------------------------------------------------------------------------
        # Initialize stations, bikes and assign cells
        # -------------------------------------------------------------------------

        # Build stations once
        stations: dict[int, Station] = {}
        for node_id, (lat, lon) in self._nodes_dict.items():
            stations[node_id] = Station(node_id, lat, lon)  # type: ignore

        # Assign cells to stations once
        for cell in self._cells.values():
            for node in cell.get_nodes():
                stations[node].set_cell(cell)

        all_stations_have_a_cell = all(
            stn.get_cell() is not None for stn in stations.values()
        )
        assert all_stations_have_a_cell, (
            "Not all stations were assigned a cell. Check cell definitions and station locations."
        )

        # Virtual station 10000
        stations[10000] = Station(10000, 0.0, 0.0)

        self._stations = stations

        # -------------------------------------------------------------------------
        # Define action and observation spaces
        # -------------------------------------------------------------------------

        # Action space
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space
        obs_dim = 17 + 2 * len(self._cells)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float16
        )

        # -------------------------------------------------------------------------
        # Initialize state variables to None/default values
        # -------------------------------------------------------------------------

        # Bike storage
        self._system_bikes: dict[int, Bike] = {}
        self._outside_system_bikes: dict[int, Bike] = {}
        self._travelling_bikes: dict[int, Bike] = {}
        self._depot = Depot()

        # Fleet parameters
        self._maximum_number_of_bikes = EnvDefaults.MAX_BIKES
        self._min_bikes_per_cell = EnvDefaults.MIN_BIKES_PER_CELL
        self._enable_repositioning = EnvDefaults.BASE_REPOSITIONING
        self._use_net_flow = EnvDefaults.NET_FLOW_BASED_REPOSITIONING

        # Station and truck objects
        self.__truck: Truck | None = None

        # Time tracking
        self._env_time = 0
        self._day = EnvDefaults.DEFAULT_DAY
        self._timeslot = EnvDefaults.DEFAULT_TIMESLOT
        self._total_timeslots = EnvDefaults.DEFAULT_TOTAL_TIMESLOTS

        # Rebalancing configuration
        self._num_rebalancing_events = EnvDefaults.DEFAULT_NUM_REBALANCING_EVENTS
        self._starting_rebalancing_event = EnvDefaults.DEFAULT_STARTING_REBALANCING_EVENT
        self._rebalancing_hours = []
        self._prediction_window = None
        self._enable_repositioning = EnvDefaults.BASE_REPOSITIONING
        self._use_net_flow = EnvDefaults.NET_FLOW_BASED_REPOSITIONING

        # Event buffer for simulation
        self._event_buffer = None

        # -------------------------------------------------------------------------
        # Initialize simulation counters
        # -------------------------------------------------------------------------
        self._timeslot = 0
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_failures = 0

        # -------------------------------------------------------------------------
        # Episode buffer state
        # -------------------------------------------------------------------------
        self._precomputed_buffers: dict[int, list[TripSample]] = {}
        self._apply_seed(seed)

        self._bg_process: BaseProcess | None = None
        self._result_queue: multiprocessing.Queue = multiprocessing.get_context("spawn").Queue(maxsize=1)

    @property
    def _truck(self) -> Truck:
        if self.__truck is None:
            raise RuntimeError(
                "Environment not initialized — call reset() before step()."
            )
        return self.__truck

    @_truck.setter
    def _truck(self, value: Truck | None) -> None:
        self.__truck = value

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Dictionary of reset options:
                - day (str): Starting day of week
                - timeslot (int): Starting timeslot (0-7)
                - total_timeslots (int): Total timeslots to simulate
                - maximum_number_of_bikes (int): Fleet size
                - minimum_number_of_bikes (int): Base bikes per cell during rebalancing
                - num_rebalancing_events (int): Number of rebalancing operations per day
                - depot_id (int): Depot cell ID

        Returns:
            Tuple of (empty_observation_dict, empty_info_dict)
        """
        if seed is not None:
            self._apply_seed(seed)
        super().reset(seed=seed)

        # ── Episode buffer management ─────────────────────────────────────────
        if not self._precomputed_buffers:
            self._precomputed_buffers = self._compute_episode_buffer(
                seed=self._episode_seed,
                first_episode=True,
                show_pbar=True,
            )
        else:
            if (
                self._bg_process is not None
                and not self._bg_process.is_alive()
                and self._result_queue.empty()
            ):
                # Process died without delivering — recompute synchronously
                self._env_logger.warning(
                    "Background result missing — recomputing synchronously."
                )
                self._precomputed_buffers = self._compute_episode_buffer(
                    seed=self._episode_seed,
                    show_pbar=True,
                )
            else:
                # Normal path (also covers: process still running but will deliver)
                if self._bg_process is not None and self._bg_process.is_alive():
                    self._env_logger.warning("Waiting for background precomputation...")
                self._precomputed_buffers = self._acquire_next_episode_buffer()

        # Parse options with defaults
        options = options or {}

        # Optional: per-episode log directory
        episode_results_path = options.get("results_path")
        if episode_results_path is not None:
            self._update_logging_path(episode_results_path)
        else:
            self._env_logger.set_enabled(False)

        # Time configuration
        self._day = options.get("day", EnvDefaults.DEFAULT_DAY)
        self._timeslot = options.get("timeslot", EnvDefaults.DEFAULT_TIMESLOT)
        self._total_timeslots = options.get(
            "total_timeslots", EnvDefaults.DEFAULT_TOTAL_TIMESLOTS
        )

        # Fleet configuration
        self._maximum_number_of_bikes = options.get(
            "maximum_number_of_bikes", self._maximum_number_of_bikes
        )
        self._min_bikes_per_cell = options.get(
            "minimum_number_of_bikes", self._min_bikes_per_cell
        )

        # Reset all cells
        for cell in self._cells.values():
            cell.reset()

        # Reset all stations
        for stn in self._stations.values():
            stn.reset()

        # Extract cell and truck configuration from options
        cell_id_list = list(self._cells.keys())
        truck_cell_id = options.get("initial_cell", self.np_random.choice(cell_id_list))
        max_truck_load = options.get("max_truck_load", EnvDefaults.MAX_TRUCK_LOAD)
        depot_id = options.get("depot_id", EnvDefaults.DEFAULT_DEPOT_ID)
        self._enable_repositioning = options.get(
            "enable_repositioning", EnvDefaults.BASE_REPOSITIONING
        )
        self._use_net_flow = options.get(
            "use_net_flow", EnvDefaults.NET_FLOW_BASED_REPOSITIONING
        )

        # Initialize depot and bike fleet
        if depot_id not in self._cells:
            raise ValueError(f"Depot cell ID {depot_id} not found in cells.")
        self._depot.id = self._cells[depot_id].get_center_node()
        self._depot.bikes = initialize_bikes(n=self._maximum_number_of_bikes)
        self._travelling_bikes = {}

        # Rebalancing configuration
        self._num_rebalancing_events = options.get(
            "num_rebalancing_events", EnvDefaults.DEFAULT_NUM_REBALANCING_EVENTS
        )

        self._starting_rebalancing_event = options.get(
            "starting_rebalancing_event", EnvDefaults.DEFAULT_STARTING_REBALANCING_EVENT
        )

        # Set rebalancing hours
        if self._num_rebalancing_events > 0:
            interval = 24 // self._num_rebalancing_events
            self._rebalancing_hours = [
                (i + self._starting_rebalancing_event) % 24 for i in range(0, 24, interval)
            ]
            self._rebalancing_hours = sorted(self._rebalancing_hours)

        self._prediction_window = (
            24 // self._num_rebalancing_events * 3600
            if self._num_rebalancing_events > 0
            else None
        )

        # -------------------------------------------------------------------------
        # Reset counters
        # -------------------------------------------------------------------------
        self._env_time = 0
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_failures = 0

        # Clear event buffers
        self._event_buffer = None

        # -------------------------------------------------------------------------
        # Initialize truck
        # -------------------------------------------------------------------------
        # Load initial bikes onto truck from depot
        initial_bike_keys = list(self._depot.bikes.keys())[
            : EnvDefaults.INITIAL_TRUCK_BIKES
        ]
        bikes = {key: self._depot.bikes.pop(key) for key in initial_bike_keys}

        truck_cell = self._cells[truck_cell_id]
        self._truck = Truck(
            position=truck_cell.get_center_node(),
            cell=truck_cell,
            bikes=bikes,
            max_load=max_truck_load,
        )

        # -------------------------------------------------------------------------
        # Initialize day/timeslot and generate events
        # -------------------------------------------------------------------------
        self._initialize_day()

        # -------------------------------------------------------------------------
        # Distribute bikes based on predicted net flow if enabled, otherwise use base repositioning
        # -------------------------------------------------------------------------
        bikes_per_cell = self._bike_repositioning(
            min_bikes=self._min_bikes_per_cell,
            max_bikes=self._maximum_number_of_bikes - self._truck.get_load(),
            enable_repositioning=self._enable_repositioning,
            use_net_flow=self._use_net_flow,
        )

        # Convert cell-based distribution to station-based
        bikes_per_station = {stn_id: 0 for stn_id in self._stations.keys()}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node = self._cells[cell_id].get_center_node()
            bikes_per_station[center_node] = num_of_bikes

        # Initialize system bikes at stations
        self._system_bikes, self._outside_system_bikes = initialize_stations(
            stations=self._stations,
            depot_bikes=self._depot.bikes,
            bikes_per_station=bikes_per_station,
        )

        info = {
            "cell_dict": self._cells,
            "nodes_dict": self._nodes_dict,
            "steps": 0,
            "failures": [],
            "depot_bikes": len(self._depot.bikes),
            "truck_bikes": self._truck.get_load(),
            "number_of_system_bikes": len(self._system_bikes),
            "number_of_outside_bikes": len(self._outside_system_bikes),
            "distance_lookup": self._distance_lookup,
        }

        # Log number of bikes in the system, in the truck, in the depot and in the outside system
        self._env_logger.set_env_time(
            convert_seconds_to_hours_minutes_day(
                day=self._day.upper(), seconds=3600 + self._env_time
            )
        )
        self._env_logger.info(
            f"Initial bikes -> "
            f"system={len(self._system_bikes)}, "
            f"truck={self._truck.get_load()}, "
            f"depot={len(self._depot.bikes)}, "
            f"outside={len(self._outside_system_bikes)}"
        )

        # ── Launch background precomputation of the next episode ─────────────
        self._start_next_episode_precomputation()

        return {}, info

    # ─────────────────────────────────────────────────────────────────────────
    # Episode buffer management
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_episode_buffer(
            self, seed: int, first_episode: bool = False, show_pbar: bool = False
    ) -> dict:
        """Synchronous wrapper — runs the worker logic in-process."""
        cache_dir = os.path.join(self._data_path, ".cache") if first_episode else None
        return compute_buffer_logic(
            seed=seed,
            data_path=self._data_path,
            global_rate_dict=self._global_rate_dict,
            distance_lookup=self._distance_lookup,
            precomputed_episode_timeslots=EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS,
            timeslots_per_day=EnvDefaults.TIMESLOTS_PER_DAY,
            timeslot_duration_seconds=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            default_day=EnvDefaults.DEFAULT_DAY,
            first_episode=first_episode,
            cache_dir=cache_dir,
            show_pbar=show_pbar,
        )

    def _start_next_episode_precomputation(self) -> None:
        """
        Spawn a background PROCESS (not thread) to compute the next episode buffer.
        The result is placed in self._result_queue when ready.
        Idempotent: does nothing if a process is already running.
        """
        if self._bg_process is not None and self._bg_process.is_alive():
            return  # Already computing, nothing to do

        # Drain any stale result from a previous (completed but unconsumed) process
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break

        next_seed = self._episode_seed + 1
        _spawn_ctx = multiprocessing.get_context("spawn")
        self._bg_process = _spawn_ctx.Process(
            target=episode_worker,
            args=(
                next_seed,
                self._data_path,
                self._global_rate_dict,
                self._distance_lookup,
                EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS,
                EnvDefaults.TIMESLOTS_PER_DAY,
                EnvDefaults.TIMESLOT_DURATION_SECONDS,
                EnvDefaults.DEFAULT_DAY,
                self._result_queue,
                True
            ),
            daemon=True,  # dies if main process dies
        )
        if self._bg_process is not None:
            self._bg_process.start()
        else:
            raise RuntimeError(
                "Background episode precomputation not started for unknown reason."
            )

    def _acquire_next_episode_buffer(self) -> dict:
        try:
            result = self._result_queue.get(timeout=300)
        except Exception:
            if self._bg_process is not None:
                self._bg_process.kill()
                self._bg_process = None
            raise RuntimeError(
                f"Background episode precomputation timed out after 300s "
                f"(seed={self._episode_seed}). Episode buffer unavailable."
            )
        if self._bg_process is not None:
            self._bg_process.join(timeout=5)
            self._bg_process = None
        # Worker may have put the exception itself into the queue on crash
        if isinstance(result, BaseException):
            raise RuntimeError(
                f"Background episode worker crashed (seed={self._episode_seed}). "
                f"Check {self._data_path}/tmp/bg_worker_crash.log for the full traceback."
            ) from result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Environment Step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step in the environment (simulate one full timeslot).

        Args:
            action: Unused (static environment has no actions).

        Returns:
            Tuple of (observation, reward, done, terminated, info)
        """
        terminated = False
        failures_this_timeslot = 0
        rebalance_times = []

        # Process events until timeslot ends
        while not terminated:
            self._env_time += EnvDefaults.STEP_DURATION_SECONDS
            self._env_logger.set_env_time(
                convert_seconds_to_hours_minutes_day(
                    day=self._day.upper(), seconds=3600 + self._env_time
                )
            )

            # Process all events that occurred before current time
            while self._event_buffer and self._event_buffer[0].time < self._env_time:
                event = self._event_buffer.popleft()

                # Process the event and get any resulting failure
                failure = event_handler(
                    event=event,
                    station_dict=self._stations,
                    nearby_nodes_dict=self._nearby_nodes_dict,
                    distance_lookup=self._distance_lookup,
                    system_bikes=self._system_bikes,
                    outside_system_bikes=self._outside_system_bikes,
                    traveling_bikes=self._travelling_bikes,
                    logger=self._env_logger,
                    logging_state_and_trips=False,
                    depot=self._depot,
                    maximum_number_of_bikes=self._maximum_number_of_bikes,
                    truck_load=self._truck.get_load(),
                )
                failures_this_timeslot += failure

            # Check for rebalancing events at hour boundaries
            if self._env_time % 3600 == 0:
                hour = ((self._env_time // 3600) + 1) % 24
                if hour in self._rebalancing_hours:
                    time_to_rebalance = self._rebalance_system()
                    rebalance_times.append(time_to_rebalance)

            # Check for timeslot termination
            timeslot_duration = EnvDefaults.TIMESLOT_DURATION_SECONDS
            if self._env_time >= timeslot_duration * (self._timeslot + 1):
                self._timeslot = (self._timeslot + 1) % EnvDefaults.TIMESLOTS_PER_DAY

                # Check for day rollover
                if self._timeslot == 0:
                    self._day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7]
                    self._days_completed += 1
                    self._initialize_day()

                terminated = True

                self._timeslots_completed += 1

        global_critic_score = 0
        for cell in self._cells.values():
            global_critic_score += cell.get_critic_score()

        if len(rebalance_times) == 0:
            rebalance_times.append(0)

        # Build info dictionary
        info = {
            "time": self._env_time + (self._timeslot * 3 + 1) * 3600,
            "day": self._day,
            "week": int(self._days_completed // 7),
            "year": self._days_completed // 365,
            "failures": failures_this_timeslot,
            "global_critic_score": global_critic_score,
            "cell_dict": self._cells,
            "depot_bikes": len(self._depot.bikes),
            "truck_bikes": self._truck.get_load(),
            "number_of_system_bikes": len(self._system_bikes),
            "number_of_outside_bikes": len(self._outside_system_bikes),
            "number_of_traveling_bikes": len(self._travelling_bikes),
            "rebalance_times": rebalance_times,
        }

        # Check if episode is complete
        done = self._timeslots_completed == self._total_timeslots

        return {}, 0, done, terminated, info

    # -------------------------------------------------------------------------
    # Rebalancing Operations
    # -------------------------------------------------------------------------

    def _rebalance_system(self) -> int:
        """
        Rebalance the system to target distribution.

        Pools used:
          - system_bikes: bikes currently at stations (or traveling)
          - depot.bikes: bikes in storage, deployable
          - outside_system_bikes: bikes on external trips, NOT touched here

        Returns:
            Time required for rebalancing in seconds.
        """
        # -----------------------------------------------------------------
        # 1. Compute TSP time BEFORE moving any bikes
        #    (_compute_rebalancing_time reads cell.get_total_bikes(),
        #    which reflects current station state — this must run first)
        # -----------------------------------------------------------------
        available_bikes = sum(
            s.get_number_of_bikes()
            for s in self._stations.values()
            if s.get_station_id() != 10000
        ) + len(self._depot.bikes)
        max_bikes = min(available_bikes, self._maximum_number_of_bikes)

        bikes_per_cell = self._bike_repositioning(
            min_bikes=1,
            max_bikes=max_bikes,
            enable_repositioning=True,
            use_net_flow=True,
        )

        self._env_logger.info(
            f"Rebalancing target — hour={((self._env_time // 3600) + 1) % 24}, "
            f"available={available_bikes}, "
            f"bikes_per_cell={dict(sorted(bikes_per_cell.items()))}, "
            f"total_targeted={sum(bikes_per_cell.values())}"
        )

        time_to_rebalance = self._compute_rebalancing_time(bikes_per_cell)

        # -----------------------------------------------------------------
        # 2. Drain all stations into a temporary pool.
        #    Only bikes currently at stations (not traveling) are moveable.
        # -----------------------------------------------------------------
        available_bikes_dict: dict[int, Bike] = {}

        for station in self._stations.values():
            if station.get_station_id() == 10000:
                continue
            while station.get_number_of_bikes() > 0:
                bike = station.unlock_bike()
                bike.set_availability(True)
                available_bikes_dict[bike.get_bike_id()] = bike

        # -----------------------------------------------------------------
        # 3. Pull all depot bikes into the pool — they are deployable
        # -----------------------------------------------------------------
        for bike_id, bike in list(self._depot.bikes.items()):
            bike.set_availability(True)
            available_bikes_dict[bike_id] = bike
        self._depot.bikes.clear()

        self._env_logger.info(
            f"Available bikes for rebalancing: {len(available_bikes_dict)} "
            f"(stations + depot)"
        )

        # -----------------------------------------------------------------
        # 4. Redistribute according to plan.
        #    Rebuild system_bikes to only contain bikes at stations.
        # -----------------------------------------------------------------

        # Remove from system_bikes only the bikes we collected (i.e. were
        # at stations). Traveling bikes remain in system_bikes untouched.
        for bike_id in available_bikes_dict:
            self._system_bikes.pop(bike_id, None)

        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node_id = self._cells[cell_id].get_center_node()
            station = self._stations[center_node_id]
            for _ in range(num_of_bikes):
                if not available_bikes_dict:
                    self._env_logger.warning(
                        f"Ran out of bikes placing cell {cell_id} — "
                        f"in-transit bikes are not yet placeable."
                    )
                    break
                bike_id = next(iter(available_bikes_dict))
                bike = available_bikes_dict.pop(bike_id)
                bike.set_battery(bike.get_max_battery())
                station.lock_bike(bike)
                self._system_bikes[bike_id] = bike

        self._env_logger.info(
            f"Total bikes in cells: {sum([cell.get_total_bikes() for cell in self._cells.values()])}"
        )

        # -----------------------------------------------------------------
        # 5. Leftover bikes (could not be placed) go back to the depot
        # -----------------------------------------------------------------
        for bike_id, bike in available_bikes_dict.items():
            bike.reset()
            self._depot.bikes[bike_id] = bike

        if self._depot.bikes:
            self._env_logger.info(
                f"{len(self._depot.bikes)} bikes returned to depot after rebalancing."
            )

        # Invariant check (debug-friendly)
        total = (
            len(self._system_bikes)
            + len(self._depot.bikes)
            + len(self._outside_system_bikes)
            + self._truck.get_load()
        )
        if total != self._maximum_number_of_bikes:
            self._env_logger.warning(
                f"Fleet invariant violated after rebalancing: "
                f"total={total}, expected={self._maximum_number_of_bikes} "
                f"(system={len(self._system_bikes)}, depot={len(self._depot.bikes)}, "
                f"outside={len(self._outside_system_bikes)}, truck={self._truck.get_load()})"
            )

        return time_to_rebalance

    def _compute_rebalancing_time(self, bikes_per_cell: dict) -> int:
        """
        Compute time required for rebalancing using TSP.

        Args:
            bikes_per_cell: Target bike distribution.

        Returns:
            Rebalancing time in seconds.
        """
        # Identify surplus and deficit nodes
        surplus_nodes = {}
        deficit_nodes = {}

        for cell_id, cell in self._cells.items():
            cell_bikes = cell.get_total_bikes()
            target_bikes = bikes_per_cell[cell_id]

            if cell_bikes > target_bikes:
                surplus_nodes[cell.get_center_node()] = cell_bikes - target_bikes
            elif cell_bikes < target_bikes:
                deficit_nodes[cell.get_center_node()] = target_bikes - cell_bikes

        # Compute TSP distance
        distance, _ = tsp_rebalancing(
            surplus_nodes, deficit_nodes, self._depot.id, self._distance_lookup
        )

        # Compute time based on distance and velocity
        hours = divmod((self._timeslot * 3 + 1) * 3600 + self._env_time, 3600)[0] % 24
        mean_truck_velocity = self._velocity_lookup[hours][self._day]
        velocity_kmh = truncated_gaussian(10, 70, mean_truck_velocity, 5)
        time_seconds = int(distance * 3.6 / velocity_kmh)

        return time_seconds

    # -------------------------------------------------------------------------
    # Event Simulation
    # -------------------------------------------------------------------------

    def _initialize_day(self) -> None:
        """
        Initialize the full day's event buffer for the current day.

        If this is a cold start (self._event_buffer is None), build a fresh
        24-hour window for the current day from precomputed buffers.

        If there are leftover events (self._event_buffer is not None), keep them,
        shift their times so that the current env_time becomes the new zero,
        then append the full new day after that.
        """

        day_index = DAYS_TO_NUM[self._day]
        first_slot_index = day_index * EnvDefaults.TIMESLOTS_PER_DAY

        if self._event_buffer is None:
            # Cold start: build full 24-hour window for current day
            day_events: list[Event] = []

            for slot_offset in range(EnvDefaults.TIMESLOTS_PER_DAY):
                slot_index = (
                    first_slot_index + slot_offset
                ) % EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS
                time_offset = slot_offset * EnvDefaults.TIMESLOT_DURATION_SECONDS

                slot_events = build_events(
                    self._precomputed_buffers[slot_index],
                    self._stations,
                    time_offset=time_offset,
                )
                day_events.extend(slot_events)

            day_events.sort(key=lambda e: e.time)
            self._event_buffer = deque(day_events)

        else:
            # Advance:
            # 1) Shift surviving events so that current env_time becomes 0
            remaining_events: list[Event] = list(self._event_buffer)
            base_time = self._env_time
            for ev in remaining_events:
                ev.time -= base_time

            # 2) Build all events for the new day, appended after leftovers
            new_day_events: list[Event] = []
            for slot_offset in range(EnvDefaults.TIMESLOTS_PER_DAY):
                slot_index = (
                    first_slot_index + slot_offset
                ) % EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS
                # Time offset starts after the leftover horizon
                time_offset = slot_offset * EnvDefaults.TIMESLOT_DURATION_SECONDS

                slot_events = build_events(
                    self._precomputed_buffers[slot_index],
                    self._stations,
                    time_offset=time_offset,
                )
                new_day_events.extend(slot_events)

            # 3) Merge leftovers and new day events into a single sorted stream
            merged = heapq.merge(remaining_events, new_day_events, key=lambda e: e.time)
            self._event_buffer = deque(merged)

            # Reset environment clock in the new frame
        self._env_time = 0

    def _load_pmf_matrix(self, day: str, timeslot: int) -> pl.DataFrame:
        """Load the PMF matrix and global rate for a given day and timeslot."""
        # Construct file path
        matrix_path = os.path.join(
            self._data_path, DEFAULT_PATHS.matrices_folder, day.lower()
        )
        pmf_matrix = pl.read_csv(
            os.path.join(matrix_path, str(timeslot).zfill(2) + "-pmf-matrix.csv")
        )

        return pmf_matrix

    # -------------------------------------------------------------------------
    # Gymnasium Interface Methods
    # -------------------------------------------------------------------------

    def _apply_seed(self, seed: int) -> None:
        """Apply seed to all RNG sources owned by this environment."""
        self._episode_seed = seed
        self.np_random, _ = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def close(self):
        if self._bg_process is not None:
            if self._bg_process.is_alive():
                self._bg_process.terminate()
            self._bg_process.join(timeout=3)
            self._bg_process.close()
            self._bg_process = None

        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue.join_thread()

        super().close()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _bike_repositioning(
        self,
        min_bikes: int,
        max_bikes: int,
        enable_repositioning: bool = False,
        use_net_flow: bool = False,
    ) -> dict:
        """
        Compute initial bike distribution.

        Args:
            enable_repositioning: If False, allocate minimum bikes per cell
                (plus uniform extras) and stop. If True, perform additional
                repositioning using either net flow or random allocation.
            use_net_flow: If True and enable_repositioning is True, bias the
                extra bikes according to predicted net outflow; otherwise
                distribute them randomly.
            max_bikes:
            min_bikes:

        Returns:
            Dictionary mapping cell_id to number of bikes.
        """
        # -------------------------------------------------------------------
        # Step 1: Uniform base allocation, adjusted if fleet is too small
        # -------------------------------------------------------------------
        available = max_bikes
        base_bikes_per_cell = min_bikes
        if base_bikes_per_cell * len(self._cells) > available:
            base_bikes_per_cell = available // len(self._cells)
            self._env_logger.warning(
                f"Base bikes per cell adjusted to {base_bikes_per_cell} "
                f"due to fleet size / truck load constraints."
            )

        bikes_per_cell = {
            cell_id: base_bikes_per_cell for cell_id in self._cells.keys()
        }
        remaining = available - base_bikes_per_cell * len(self._cells)

        self._env_logger.info(
            f"Base bikes per cell adjusted to {base_bikes_per_cell} "
            f"with {len(self._cells)} cells"
        )

        if remaining == 0:
            return bikes_per_cell

        if remaining < 0:
            raise ValueError(
                "Fleet size minus truck load is too small to allocate "
                "the base number of bikes per cell."
            )

        # If repositioning is disabled, just keep uniform allocation
        if not enable_repositioning:
            return bikes_per_cell

        # -------------------------------------------------------------------
        # Step 2: Net flow biased distribution (only current timeslot events)
        # -------------------------------------------------------------------
        bikes_positioned = 0

        if use_net_flow:
            net_flow_per_cell = {cell_id: 0 for cell_id in self._cells.keys()}

            for event in self._event_buffer:
                if event.time > EnvDefaults.TIMESLOT_DURATION_SECONDS:
                    break  # buffer is sorted — stop at next-timeslot events
                if event.is_departure():
                    station_id = event.trip.get_start_location().get_station_id()
                    if station_id != 10000:
                        cell = self._stations[station_id].get_cell()
                        if cell is None:
                            raise ValueError(
                                f"Departure event with invalid cell for station {station_id}"
                            )
                        net_flow_per_cell[cell.get_id()] -= 1
                elif event.is_arrival():
                    station_id = event.trip.get_end_location().get_station_id()
                    if station_id != 10000:
                        cell = self._stations[station_id].get_cell()
                        if cell is None:
                            raise ValueError(
                                f"Arrival event with invalid cell for station {station_id}"
                            )
                        net_flow_per_cell[cell.get_id()] += 1

            total_negative_flow = sum(f for f in net_flow_per_cell.values() if f < 0)

            if total_negative_flow < 0:  # guard: at least one cell has net outflow
                for cell_id, flow in net_flow_per_cell.items():
                    if flow < 0:
                        proportional_bikes = int(
                            (flow / total_negative_flow) * remaining
                        )
                        bikes_per_cell[cell_id] += proportional_bikes
                        bikes_positioned += proportional_bikes

        # -------------------------------------------------------------------
        # Step 3: Distribute any remaining bikes randomly
        # -------------------------------------------------------------------
        leftover = remaining - bikes_positioned
        if leftover > 0:
            cell_ids = list(self._cells.keys())
            chosen = self.np_random.choice(cell_ids, size=leftover, replace=True)
            for cell_id in chosen:
                bikes_per_cell[cell_id] += 1

        return bikes_per_cell

    def _update_logging_path(self, results_path: str | None) -> None:
        """
        Update the env logging directory at runtime (e.g. per episode).
        """
        self._results_path = results_path
        if results_path is None:
            return

        episode_log_dir = os.path.join(results_path, "logs")
        self._env_logger.reconfigure(
            log_dir=episode_log_dir,
            filename="env.log",
        )
