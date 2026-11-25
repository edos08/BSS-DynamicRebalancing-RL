"""
Fully Dynamic Bike-Sharing System Environment for Reinforcement Learning.

This module implements a Gymnasium environment for training RL agents to perform
dynamic bike rebalancing in a bike-sharing system. The environment simulates
bike trips, truck movements, and the state of the entire network.

Author: Edoardo Scarpel
"""

import heapq
import logging
import math
import multiprocessing
import os.path

import gymnasium as gym
import numpy as np
import osmnx as ox
import polars as pl

from collections import deque
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator import Bike
from gymnasium_env.simulator.bike_simulator import build_events, event_handler
from gymnasium_env.simulator.env_logger import EnvLogger
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import TripSample
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.truck_simulator import (
    ACTION_TO_DIRECTION,
    charge_bike,
    drop_bike,
    move,
    pick_up_bike,
    stay,
)
from gymnasium_env.simulator.utils import (
    DAYS_TO_NUM,
    DEFAULT_PATHS,
    NUM_TO_DAYS,
    Actions,
    compute_buffer_logic,
    convert_seconds_to_hours_minutes_day,
    detect_self_loops,
    episode_worker,
    initialize_bikes,
    initialize_graph,
    initialize_stations,
    load_preprocessed_data,
)

# =============================================================================
# Module-Level Configuration
# =============================================================================

# Enable detailed logging of state and trips (typically disabled for training)
logging_state_and_trips = False


# =============================================================================
# Environment Constants
# =============================================================================


class EnvDefaults:
    """Default configuration values for the environment."""

    # Bike fleet parameters
    MAX_BIKES = 1000
    MIN_BIKES_PER_CELL = 5
    BASE_REPOSITIONING = True
    NET_FLOW_BASED_REPOSITIONING = False

    # Truck parameters
    MAX_TRUCK_LOAD = 30
    INITIAL_TRUCK_BIKES = 15

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

    # Precomputation parameters
    PRECOMPUTED_EPISODE_TIMESLOTS = 56  # 7 days × 8 slots

    # Other parameters
    USE_BINARY_CRITIC = True

    # How the critic score anticipates future demand:
    #   "poisson": forecast sampled from the per-slot Poisson rates, independent of the
    #              trips the simulator will actually generate (no look-ahead)
    #   "oracle":  reads the realised future trips from the simulator's event buffer
    CRITIC_MODE = "poisson"
    CRITIC_MODES = ("poisson", "oracle")
    FORECAST_STREAM = 0xF0CA57  # extra entropy so the forecast RNG is independent of trip sampling


class BorderType:
    """Cell border classification based on number of missing adjacent cells."""

    NORMAL = 0  # All 4 adjacent cells exist
    EDGE = 1  # 1 adjacent cell missing
    CORNER = 2  # 2 adjacent cells missing
    DEAD_END = 3  # 3 adjacent cells missing


class RewardComponents:
    """Reward function component values."""

    # Base step cost
    BASE_COST = -0.1

    # Invalid action penalty
    INVALID_ACTION = -1.0

    # Loop detection penalty
    LOOP_PENALTY = -0.6

    # Drop bike rewards
    DROP_BASE = 0.01
    DROP_REBALANCED_CRITICAL = 2.0
    DROP_IN_CRITICAL = 1.0
    DROP_IN_SURPLUS = -0.5

    # Pick-up rewards
    PICKUP_FROM_CRITICAL = -0.5
    PICKUP_UNBALANCED_CELL = -2.0
    PICKUP_FROM_SURPLUS = 0.2

    # Charge bike rewards
    CHARGE_USELESS_CRITICAL = -0.1
    CHARGE_USELESS_NORMAL = -0.3
    CHARGE_USEFUL_CRITICAL = 0.5
    CHARGE_USEFUL_NORMAL = 0.3
    CHARGE_LOW_BATTERY_THRESHOLD = 0.8

    # Eligibility penalties
    ELIGIBILITY_HIGH_THRESHOLD = 0.7
    ELIGIBILITY_LOW_THRESHOLD = 0.2
    ELIGIBILITY_REVISIT_PENALTY = -0.2
    ELIGIBILITY_EXPLORATION_BONUS = 0.3
    ELIGIBILITY_EMPTY_TRUCK_PENALTY = -0.05

    # Stay penalties
    STAY_BASE = -0.1
    STAY_IN_CRITICAL = -1.0
    STAY_NO_CRITIC = 0.0

    # Other
    SURPLUS_THRESHOLD = -0.67
    DEPLOY_WEIGHT = 0.02
    DEPOT_WEIGHT = 0.02


# =============================================================================
# Auxiliary classes
# =============================================================================


@dataclass
class Depot:
    id: int | None = None
    bikes: dict[int, Bike] = field(default_factory=dict)


# =============================================================================
# Main Environment Class
# =============================================================================


class FullyDynamicEnv(gym.Env):
    """
    A Gymnasium environment for bike-sharing system dynamic rebalancing.

    This environment simulates a bike-sharing network where a rebalancing truck
    must distribute bikes across stations to minimize service failures. The agent
    controls the truck's movements and bike pickup/drop operations.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        data_path: str,
        results_path: str,
        seed: int = 42,
        logging_enabled: bool = False,
    ):
        """
        Initialize the bike-sharing environment.

        Args:
            data_path: Path to directory containing network data, PMF matrices, etc.
            results_path: Path to directory for storing logs and results.
        """
        super().__init__()

        # Store paths and initialize logger
        self._data_path = data_path
        self._results_path = results_path

        # Env logger: lazy init
        log_dir = (
            os.path.join(results_path, "logs") if results_path is not None else None
        )
        self._env_logger = EnvLogger(name="env")
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

        # RL parameters
        self._discount_factor = EnvDefaults.DISCOUNT_FACTOR
        self._eligibility_decay = EnvDefaults.ELIGIBILITY_DECAY
        self._borders_eligibility_decay = EnvDefaults.BORDER_ELIGIBILITY_DECAY
        self._global_critic_score = 0.0
        self._use_binary_critic = EnvDefaults.USE_BINARY_CRITIC
        self._reward_params = None

        # Action tracking
        self._last_move_action = 0
        self._invalid_action = False
        self._last_cell_border_type = BorderType.NORMAL

        # Event buffers
        self._event_buffer = None

        # Demand forecast used by the critic score (critic_mode="poisson"):
        # per cell, sorted arrays of future event times (s, relative to the current slot start)
        # and deltas (+1 departure, -1 arrival).
        self._critic_mode = EnvDefaults.CRITIC_MODE
        self._cell_order = list(self._cells.keys())
        self._forecast_times: list[np.ndarray] | None = None
        self._forecast_deltas: list[np.ndarray] | None = None
        self._slot_rate_cache: dict[tuple[str, int], dict] = {}

        # -------------------------------------------------------------------------
        # Initialize simulation counters
        # -------------------------------------------------------------------------
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_visits = 1
        self._total_failures = 0
        self._total_invalid_actions = 0

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

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:  # type: ignore
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Dictionary of reset options:
                - logging (bool): Enable detailed logging
                - day (str): Starting day of week
                - timeslot (int): Starting timeslot (0-7)
                - total_timeslots (int): Total timeslots to simulate
                - maximum_number_of_bikes (int): Fleet size
                - min_bikes_per_cell (int): Minimum bikes per cell
                - depot_id (int): Depot cell ID
                - initial_cell (int): Truck starting cell
                - max_truck_load (int): Truck capacity
                - discount_factor (float): RL discount factor
                - reward_params (dict): Custom reward parameters
                - critic_mode (str): "poisson" (default, forecast sampled from the
                  demand rates) or "oracle" (reads the realised future trips)

        Returns:
            Tuple of (initial_observation, info_dict)
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
            if self._result_queue.empty() and (
                self._bg_process is None or not self._bg_process.is_alive()
            ):
                # Process died without delivering — recompute synchronously
                self._env_logger.warning(
                    "Background result missing — recomputing synchronously."
                )
                self._precomputed_buffers = self._compute_episode_buffer(
                    self._episode_seed
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

        # RL configuration
        self._discount_factor = options.get(
            "discount_factor", EnvDefaults.DISCOUNT_FACTOR
        )
        self._reward_params = options.get("reward_params", None)

        # Reset action state
        self._invalid_action = False
        self._global_critic_score = 0.0
        self._use_binary_critic = options.get(
            "use_binary_critic", EnvDefaults.USE_BINARY_CRITIC
        )
        self._critic_mode = options.get("critic_mode", EnvDefaults.CRITIC_MODE)
        if self._critic_mode not in EnvDefaults.CRITIC_MODES:
            raise ValueError(
                f"critic_mode must be one of {EnvDefaults.CRITIC_MODES}, got {self._critic_mode!r}"
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

        # Mark initial truck cell as visited
        self._cells[truck_cell_id].set_visits(1)

        # Initialize depot and bike fleet
        if depot_id not in self._cells:
            raise ValueError(f"Depot cell ID {depot_id} not found in cells.")
        self._depot.id = self._cells[depot_id].get_center_node()
        self._depot.bikes = initialize_bikes(n=self._maximum_number_of_bikes)
        self._travelling_bikes = {}

        # -------------------------------------------------------------------------
        # Reset counters
        # -------------------------------------------------------------------------
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_visits = 1
        self._total_failures = 0
        self._total_invalid_actions = 0

        # Clear event buffers
        self._event_buffer = None
        self._forecast_times = None
        self._forecast_deltas = None

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
        self._initialize_day_timeslot()

        # -------------------------------------------------------------------------
        # Distribute bikes based on predicted net flow if enabled, otherwise use base repositioning
        # -------------------------------------------------------------------------
        bikes_per_cell = self._bike_repositioning(
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

        # -------------------------------------------------------------------------
        # Update graph with initial metrics
        # -------------------------------------------------------------------------
        self._update_cells_metrics()

        # -------------------------------------------------------------------------
        # Build initial observation and info
        # -------------------------------------------------------------------------
        observation = self._get_obs()
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
        self._env_logger.info(
            f"Initial bikes -> "
            f"system={len(self._system_bikes)}, "
            f"truck={self._truck.get_load()}, "
            f"depot={len(self._depot.bikes)}, "
            f"outside={len(self._outside_system_bikes)}"
        )

        # ── Launch background precomputation of the next episode ─────────────
        self._start_next_episode_precomputation()

        return observation, info

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

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:  # type: ignore
        """
        Execute one step in the environment.

        Args:
            action: Integer action from the action space.

        Returns:
            Tuple of (observation, reward, done, terminated, info)
        """
        # Execute the action
        self._invalid_action = False

        # Calculate current mean truck velocity
        hours = divmod((self._timeslot * 3 + 1) * 3600 + self._env_time, 3600)[0] % 24
        mean_truck_velocity = self._velocity_lookup[hours][self._day]

        elapsed_time = 0
        distance = 0

        if action == Actions.STAY.value:
            elapsed_time = stay(truck=self._truck)

        elif action in ACTION_TO_DIRECTION:
            elapsed_time, distance, self._invalid_action = move(
                action=action,
                truck=self._truck,
                distance_lookup=self._distance_lookup,
                cell_dict=self._cells,
                mean_velocity=mean_truck_velocity,
            )

        elif action == Actions.DROP_BIKE.value:
            elapsed_time, distance, self._invalid_action = drop_bike(
                truck=self._truck,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes,
                maximum_number_of_bikes=self._maximum_number_of_bikes,
            )

        elif action == Actions.PICK_UP_BIKE.value:
            elapsed_time, distance, self._invalid_action = pick_up_bike(
                truck=self._truck,
                station_dict=self._stations,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes,
            )

        elif action == Actions.CHARGE_BIKE.value:
            elapsed_time, distance, self._invalid_action = charge_bike(
                truck=self._truck,
                station_dict=self._stations,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes,
            )

        # Start new log line
        timeslot_index = 8 * DAYS_TO_NUM[self._day] + self._timeslot
        self._env_logger.new_log_line(timeslot=timeslot_index)
        self._env_logger.log_starting_action(
            action=str(Actions(action).name),
            t=elapsed_time,
            cell_id=self._truck.get_cell().get_id(),
            invalid=False,
        )

        # Store pre-update state for reward calculation
        truck_cell = self._truck.get_cell()
        old_eligibility_score = truck_cell.get_eligibility_score()
        old_critic_score = truck_cell.get_critic_score()
        old_global_critic_score = self._global_critic_score
        old_surplus_bikes = truck_cell.get_surplus_bikes()

        self._env_logger.info(
            f"Cell {truck_cell.get_id()} before update has "
            f"Eligibility: {old_eligibility_score} Critic: {old_critic_score}"
        )

        # -------------------------------------------------------------------------
        # Advance simulation
        # -------------------------------------------------------------------------
        # Process events during elapsed time
        steps = math.ceil(elapsed_time / EnvDefaults.STEP_DURATION_SECONDS)
        failures = self._jump_to_next_state(steps)

        # Handle bike placement after movement
        is_placement_action = action in {
            Actions.CHARGE_BIKE.value,
            Actions.DROP_BIKE.value,
        }

        if is_placement_action and not self._invalid_action:
            station = self._stations[self._truck.get_position()]
            bike = self._truck.unload_bike()
            station.lock_bike(bike)
            self._system_bikes[bike.get_bike_id()] = bike

        # Final state update -> S'
        failures.extend(self._jump_to_next_state(steps=1))
        steps += 1
        self._total_failures += sum(failures)

        # Update cell statistics
        self._update_cells_metrics()
        truck_cell.set_eligibility_score(1.0)

        self._env_logger.log_state(
            step=steps,
            time=convert_seconds_to_hours_minutes_day(
                day=self._day.upper(),
                seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time,
            ),
        )
        self._env_logger.log_truck(
            truck=self._truck,
            depot_bikes=len(self._depot.bikes),
        )

        if not self._invalid_action:
            truck_cell.set_ops(truck_cell.get_ops() + 1)
            if action == Actions.PICK_UP_BIKE.value:
                truck_cell.set_pick_ups(truck_cell.get_pick_ups() + 1)
            elif action == Actions.DROP_BIKE.value:
                truck_cell.set_drops(truck_cell.get_drops() + 1)

        if action in {
            Actions.UP.value,
            Actions.DOWN.value,
            Actions.LEFT.value,
            Actions.RIGHT.value,
        }:
            truck_cell.set_visits(truck_cell.get_visits() + 1)
            self._total_visits += 1

        # Update action tracking
        if not self._invalid_action:
            self._last_move_action = action
        else:
            self._total_invalid_actions += 1

        self._last_cell_border_type = sum(self._get_borders())

        self._env_logger.log_ending_action(
            invalid=self._invalid_action,
            time=convert_seconds_to_hours_minutes_day(
                day=self._day.upper(),
                seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time,
            ),
        )

        # Compute outputs
        reward = self._get_reward(
            action,
            old_eligibility_score,
            old_critic_score,
            old_global_critic_score,
            old_surplus_bikes,
        )
        observation = self._get_obs(action)

        # Build info dictionary
        info = {
            "time": self._env_time + (self._timeslot * 3 + 1) * 3600,
            "day": self._day
            if self._timeslot < EnvDefaults.TIMESLOTS_PER_DAY - 1
            else NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7],
            "week": self._days_completed // 7,
            "year": self._days_completed // 365,
            "failures": failures,
            "steps": steps,
            "global_critic_score": self._global_critic_score,
            "total_invalid_actions": self._total_invalid_actions,
            "cell_dict": self._cells,
            "distance": distance,
            "depot_bikes": len(self._depot.bikes),
            "truck_bikes": self._truck.get_load(),
            "number_of_system_bikes": len(self._system_bikes),
            "number_of_outside_bikes": len(self._outside_system_bikes),
            "number_of_traveling_bikes": len(self._travelling_bikes),
        }

        # Check for timeslot/episode termination
        terminated, done = self._check_termination()

        # Apply final penalty if episode complete
        if done:
            reward -= self._total_failures / self._total_timeslots  # / 10.0
            self._env_logger.log_done(
                time=convert_seconds_to_hours_minutes_day(
                    day=self._day.upper(),
                    seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time,
                )
            )

        return observation, reward, done, terminated, info

    def _check_termination(self) -> tuple[bool, bool]:
        """
        Check if timeslot or episode has ended.

        Returns:
            Tuple of (terminated, done) flags.
        """
        terminated = False

        # Check if current timeslot is complete
        if self._env_time >= EnvDefaults.TIMESLOT_DURATION_SECONDS:
            # Advance to next timeslot
            self._timeslot = (self._timeslot + 1) % EnvDefaults.TIMESLOTS_PER_DAY

            # Advance day if needed
            if self._timeslot == 0:
                self._day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7]
                self._days_completed += 1

            # Preserve remaining time into next timeslot
            env_time_diff = self._env_time - EnvDefaults.TIMESLOT_DURATION_SECONDS
            self._initialize_day_timeslot()
            self._env_time = env_time_diff

            self._timeslots_completed += 1
            terminated = True

        # Check if episode is complete
        done = self._timeslots_completed == self._total_timeslots

        return terminated, done

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium Interface Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_seed(self, seed: int) -> None:
        """Apply seed to all RNG sources owned by this environment."""
        self._episode_seed = seed
        self.np_random, _ = seeding.np_random(seed)
        self._forecast_rng = np.random.default_rng([seed, EnvDefaults.FORECAST_STREAM])
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

    # ─────────────────────────────────────────────────────────────────────────
    # Event Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_day_timeslot(self) -> None:
        current_slot_index = (
            DAYS_TO_NUM[self._day] * EnvDefaults.TIMESLOTS_PER_DAY + self._timeslot
        )
        next_slot_index = (
            current_slot_index + 1
        ) % EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS

        if self._event_buffer is None:
            # Cold start: build full 6-hour window
            current_events = build_events(
                self._precomputed_buffers[current_slot_index],
                self._stations,
                time_offset=0,
            )
            next_events = build_events(
                self._precomputed_buffers[next_slot_index],
                self._stations,
                time_offset=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            )
            self._event_buffer = deque(
                heapq.merge(current_events, next_events, key=lambda e: e.time)
            )
        else:
            # Advance: shift surviving events, append next slot fresh
            for event in self._event_buffer:
                event.time -= EnvDefaults.TIMESLOT_DURATION_SECONDS

            next_events = build_events(
                self._precomputed_buffers[next_slot_index],
                self._stations,
                time_offset=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            )
            self._event_buffer = deque(
                heapq.merge(self._event_buffer, next_events, key=lambda e: e.time)
            )

        # --- Demand/arrival rates of the current slot (stations and cells) ---
        station_dep, station_arr = self._slot_station_rates(self._day, self._timeslot)
        for stn_id, stn in self._stations.items():
            stn.set_request_rate(station_dep[stn_id])
            stn.set_arrival_rate(station_arr.get(stn_id, 0.0))

        for cell in self._cells.values():
            nodes = cell.get_nodes()
            cell.set_demand_rate(
                sum(self._stations[n].get_demand_rate() for n in nodes)
            )
            cell.set_arrival_rate(
                sum(self._stations[n].get_arrival_rate() for n in nodes)
            )

        # --- Demand forecast for the critic score ---
        if self._critic_mode == "poisson":
            self._advance_forecast(next_slot_index)

        # Reset environment clock
        self._env_time = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Demand rates and forecast (critic score)
    # ─────────────────────────────────────────────────────────────────────────

    def _slot_station_rates(
        self, day: str, timeslot: int
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Per-station request and arrival rates (trips/s) of a (day, timeslot); cached."""
        key = (day.lower(), timeslot)
        entry = self._slot_rate_cache.get(key)
        if entry is None:
            pmf = self._load_pmf_matrix(day, timeslot)
            global_rate = self._global_rate_dict[key]
            values = pmf.drop("node_id")
            dep = dict(
                zip(
                    pmf["node_id"].to_list(),
                    (values.sum_horizontal() * global_rate).to_list(),
                )
            )
            arr = {
                int(col): total * global_rate
                for col, total in zip(values.columns, values.sum().row(0))
            }
            entry = {"station": (dep, arr)}
            self._slot_rate_cache[key] = entry
        return entry["station"]

    def _slot_cell_rates(self, day: str, timeslot: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell (departure, arrival) rates in trips/s, ordered as self._cell_order; cached."""
        self._slot_station_rates(day, timeslot)
        entry = self._slot_rate_cache[(day.lower(), timeslot)]
        if "cell" not in entry:
            dep, arr = entry["station"]
            entry["cell"] = (
                np.array(
                    [sum(dep[n] for n in self._cells[c].get_nodes()) for c in self._cell_order]
                ),
                np.array(
                    [sum(arr.get(n, 0.0) for n in self._cells[c].get_nodes()) for c in self._cell_order]
                ),
            )
        return entry["cell"]

    def _sample_forecast_slot(
        self, day: str, timeslot: int, offset: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Sample the trips of one slot from the Poisson model (per cell: a Poisson number of
        departures and of arrivals, uniform in time). Independent of the simulator's events.
        """
        duration = EnvDefaults.TIMESLOT_DURATION_SECONDS
        dep_rate, arr_rate = self._slot_cell_rates(day, timeslot)
        n_dep = self._forecast_rng.poisson(dep_rate * duration)
        n_arr = self._forecast_rng.poisson(arr_rate * duration)
        samples = []
        for i in range(len(self._cell_order)):
            k = int(n_dep[i] + n_arr[i])
            times = offset + self._forecast_rng.uniform(0.0, duration, size=k)
            deltas = np.concatenate(
                [np.ones(n_dep[i], dtype=np.int8), -np.ones(n_arr[i], dtype=np.int8)]
            )
            samples.append((times, deltas))
        return samples

    def _extend_forecast(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for i, (times, deltas) in enumerate(samples):
            merged_t = np.concatenate([self._forecast_times[i], times])
            merged_d = np.concatenate([self._forecast_deltas[i], deltas])
            order = np.argsort(merged_t, kind="stable")
            self._forecast_times[i] = merged_t[order]
            self._forecast_deltas[i] = merged_d[order]

    def _advance_forecast(self, next_slot_index: int) -> None:
        """
        Keep the forecast window aligned with the event buffer: [current slot, next slot].
        At a slot change the already drawn scenario of the upcoming slot becomes the
        current one and only the new "next" slot is sampled.
        """
        duration = EnvDefaults.TIMESLOT_DURATION_SECONDS
        next_day = NUM_TO_DAYS[next_slot_index // EnvDefaults.TIMESLOTS_PER_DAY]
        next_timeslot = next_slot_index % EnvDefaults.TIMESLOTS_PER_DAY

        if self._forecast_times is None:
            empty_t = [np.empty(0) for _ in self._cell_order]
            empty_d = [np.empty(0, dtype=np.int8) for _ in self._cell_order]
            self._forecast_times, self._forecast_deltas = empty_t, empty_d
            self._extend_forecast(self._sample_forecast_slot(self._day, self._timeslot, 0.0))
        else:
            for i in range(len(self._cell_order)):
                times = self._forecast_times[i] - duration
                keep = times >= 0.0
                self._forecast_times[i] = times[keep]
                self._forecast_deltas[i] = self._forecast_deltas[i][keep]
        self._extend_forecast(self._sample_forecast_slot(next_day, next_timeslot, duration))

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

    def _jump_to_next_state(self, steps: int = 0) -> list:
        """Advance the simulation by the given number of steps."""
        failures = []

        for step in range(steps):
            # Increment environment time
            self._env_time += EnvDefaults.STEP_DURATION_SECONDS
            self._env_logger.set_env_time(
                convert_seconds_to_hours_minutes_day(
                    day=self._day.upper(),
                    seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time,
                )
            )

            # Update eligibility scores for all cells
            for _, cell in self._cells.items():
                if cell.has_all_neighbors():
                    cell.update_eligibility_score(self._eligibility_decay)
                else:
                    cell.update_eligibility_score(self._borders_eligibility_decay)

            # Process events that occurred during this step
            step_failures = self._process_events()
            failures.append(step_failures)

        return failures

    def _process_events(self) -> int:
        """
        Process all events that occurred before current env_time.

        Returns:
            Number of failures during this step.
        """
        total_step_failures = 0

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
                logging_state_and_trips=logging_state_and_trips,
                depot=self._depot,
                maximum_number_of_bikes=self._maximum_number_of_bikes,
                truck_load=self._truck.get_load(),
            )
            total_step_failures += failure

        return total_step_failures

    # ─────────────────────────────────────────────────────────────────────────
    # Observation Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self, action: int = None) -> np.array:  # type: ignore
        """Construct the observation vector."""
        # Previous action encoding
        ohe_previous_action = np.array(
            [1.0 if action == a.value else 0.0 for a in Actions], dtype=np.float16
        )

        # Truck cell position encoding
        truck_cell_id = self._truck.get_cell().get_id()
        sorted_cells_keys = self._sorted_cell_ids

        ohe_cell_position = np.array(
            [1.0 if truck_cell_id == cell_id else 0.0 for cell_id in sorted_cells_keys],
            dtype=np.float16,
        )

        # Critical cells encoding
        ohe_cell_critic = np.array(
            [1.0 if cell.is_critical() else 0.0 for cell in self._cells.values()],
            dtype=np.float16,
        )

        # Border flags
        ohe_borders = self._get_borders()

        # Scalar features
        truck_cell = self._truck.get_cell()

        depot_load_ratio = len(self._depot.bikes) / self._maximum_number_of_bikes
        system_ratio = (
            len(self._system_bikes) + self._truck.get_load()
        ) / self._maximum_number_of_bikes

        scalar_features = np.array(
            [
                system_ratio,
                depot_load_ratio,
                1.0 if truck_cell.get_surplus_bikes() > 0 else 0.0,  # Surplus flag
                1.0 if truck_cell.get_total_bikes() == 0 else 0.0,  # Empty cell flag
                self._global_critic_score / len(self._cells),  # Global critic score
            ],
            dtype=np.float16,
        )

        # Concatenate all features
        observation = np.concatenate(
            [
                scalar_features,
                ohe_previous_action,
                ohe_cell_position,
                ohe_cell_critic,
                ohe_borders,
            ]
        )

        return observation

    def _get_borders(self) -> np.array:  # type: ignore
        """
        Get border flags indicating which adjacent cells are missing.

        Returns:
            Array of 4 flags (one per direction).
        """
        adjacent_cells = self._truck.get_cell().get_adjacent_cells()
        return np.array(
            [1 if adj_cell is None else 0 for adj_cell in adjacent_cells.values()],
            dtype=np.float16,
        )

    def _get_truck_position(self) -> tuple[float, float]:
        """
        Get the truck's normalized position coordinates.

        Returns:
            Tuple of (normalized_lat, normalized_lon).
        """
        truck_coords = self._nodes_dict[self._truck.get_position()]

        normalized_lat = (truck_coords[0] - self._min_lat) / (
            self._max_lat - self._min_lat
        )
        normalized_lon = (truck_coords[1] - self._min_lon) / (
            self._max_lon - self._min_lon
        )

        return normalized_lat, normalized_lon

    # ─────────────────────────────────────────────────────────────────────────
    # Reward Function
    # ─────────────────────────────────────────────────────────────────────────

    def _get_reward(
        self,
        action: int,
        old_eligibility_score: float,
        old_critic_score: float,
        old_global_critic_score: float,
        old_surplus_bikes: float,
    ) -> float:
        """
        Compute the reward for the current step.

        Args:
            action: Action taken this step.
            old_eligibility_score: Eligibility score before action.
            old_critic_score: Critic score before action.
            old_global_critic_score: Global critic score before action.

        Returns:
            Float reward value.
        """
        # Handle invalid actions immediately
        if self._invalid_action:
            return RewardComponents.INVALID_ACTION

        # Compute common state indicators
        loop_detected = detect_self_loops((action, self._last_move_action))
        was_critical = old_critic_score > 0.0
        was_surplus = old_surplus_bikes > 0
        # was_surplus = old_critic_score <= RewardComponents.SURPLUS_THRESHOLD
        is_critical = self._truck.get_cell().is_critical()

        # Initialize reward components
        drop_reward = 0.0
        pick_up_reward = 0.0
        bike_charge_reward = 0.0
        eligibility_penalty = 0.0
        stay_penalty = 0.0
        loop_penalty = 0.0

        # Apply loop penalty
        if loop_detected:
            loop_penalty = RewardComponents.LOOP_PENALTY

        # Compute action-specific rewards
        if action == Actions.DROP_BIKE.value:
            drop_reward = self._compute_drop_reward(
                was_critical, is_critical, was_surplus
            )

        elif action == Actions.PICK_UP_BIKE.value:
            pick_up_reward = self._compute_pickup_reward(
                was_critical, is_critical, was_surplus, loop_detected
            )

        elif action == Actions.CHARGE_BIKE.value:
            bike_charge_reward = self._compute_charge_reward(was_critical)
            # bike_charge_reward = 0.0  # Placeholder for future implementation

        elif action in {
            Actions.UP.value,
            Actions.DOWN.value,
            Actions.LEFT.value,
            Actions.RIGHT.value,
        }:
            eligibility_penalty = self._compute_movement_penalty(
                old_eligibility_score, was_critical
            )

        elif action == Actions.STAY.value:
            stay_penalty, loop_penalty = self._compute_stay_penalty(
                was_critical, old_global_critic_score, loop_penalty
            )

        delta_global = self._global_critic_score - old_global_critic_score

        deploy_ratio = len(self._system_bikes) / self._maximum_number_of_bikes
        r_deploy = RewardComponents.DEPLOY_WEIGHT * deploy_ratio

        idle_ratio = len(self._depot.bikes) / self._maximum_number_of_bikes
        r_depot = -RewardComponents.DEPOT_WEIGHT * idle_ratio

        # Combine all reward components
        reward = (
            RewardComponents.BASE_COST
            + drop_reward
            + pick_up_reward
            + bike_charge_reward
            + eligibility_penalty
            + stay_penalty
            + loop_penalty
            + r_deploy
            + r_depot
            - 0.5 * delta_global
        )

        self._env_logger.info(
            f"split reward {drop_reward} {pick_up_reward} {bike_charge_reward} "
            f"{eligibility_penalty} {stay_penalty} {loop_penalty}"
        )

        return reward

    # ─────────────────────────────────────────────────────────────────────────
    # Reward Computation Utility Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_drop_reward(
        self, was_critical: bool, is_critical: bool, was_surplus: bool
    ) -> float:
        """Compute reward for drop bike action."""
        drop_reward = RewardComponents.DROP_BASE

        if (
            was_critical and not is_critical
        ):  # If the truck manages to rebalance the cell -> Big reward and (optional) reset all eligibility to 0.0
            # Successfully rebalanced a critical cell
            drop_reward = RewardComponents.DROP_REBALANCED_CRITICAL
            self._truck.get_cell().update_rebalanced_times()
            self._reset_eligibility_scores()
        elif was_critical:
            # Dropped in critical cell but still critical
            drop_reward = RewardComponents.DROP_IN_CRITICAL
        elif was_surplus:
            # Dropped in surplus cell (bad)
            drop_reward = RewardComponents.DROP_IN_SURPLUS

        return drop_reward

    @staticmethod
    def _compute_pickup_reward(
        was_critical: bool, is_critical: bool, was_surplus: bool, loop_detected: bool
    ) -> float:
        """Compute reward for pickup bike action."""
        pick_up_reward = 0.0

        if was_critical:
            pick_up_reward = RewardComponents.PICKUP_FROM_CRITICAL
        elif is_critical and not was_critical:
            # Unbalanced a cell by picking up
            pick_up_reward = RewardComponents.PICKUP_UNBALANCED_CELL
        elif was_surplus and not loop_detected:
            pick_up_reward = RewardComponents.PICKUP_FROM_SURPLUS

        return pick_up_reward

    def _compute_charge_reward(self, was_critical: bool) -> float:
        """Compute reward for charge bike action."""
        # Check if charge was useful (bike was sufficiently discharged)
        if self._truck.last_charge < RewardComponents.CHARGE_LOW_BATTERY_THRESHOLD:
            # Bike was already nearly full → useless charge
            if was_critical:
                return RewardComponents.CHARGE_USELESS_CRITICAL  # -0.1
            else:
                return RewardComponents.CHARGE_USELESS_NORMAL  # -0.3
        else:
            # Bike was sufficiently discharged → useful charge
            if was_critical:
                return RewardComponents.CHARGE_USEFUL_CRITICAL  # +0.5
            else:
                return RewardComponents.CHARGE_USEFUL_NORMAL  # +0.3

    def _compute_movement_penalty(
        self, old_eligibility_score: float, was_critical: bool
    ) -> float:
        """Compute penalty for movement actions based on eligibility."""
        eligibility_penalty = 0.0

        if (
            old_eligibility_score > RewardComponents.ELIGIBILITY_HIGH_THRESHOLD
            and not was_critical
        ):
            # Revisiting a recently visited cell
            eligibility_penalty = RewardComponents.ELIGIBILITY_REVISIT_PENALTY
            # Bonus for exiting border cells
            eligibility_penalty += 0.1 * self._last_cell_border_type
        elif old_eligibility_score < RewardComponents.ELIGIBILITY_LOW_THRESHOLD:
            # Exploring unvisited areas
            eligibility_penalty += RewardComponents.ELIGIBILITY_EXPLORATION_BONUS
            # Penalty when truck is empty
        if self._truck.get_load() == 0:
            eligibility_penalty += RewardComponents.ELIGIBILITY_EMPTY_TRUCK_PENALTY

        return eligibility_penalty

    @staticmethod
    def _compute_stay_penalty(
        was_critical: bool, old_global_critic_score: float, loop_penalty: float
    ) -> tuple[float, float]:
        """Compute penalty for stay action."""
        stay_penalty = RewardComponents.STAY_BASE

        if was_critical:
            stay_penalty = RewardComponents.STAY_IN_CRITICAL
        elif old_global_critic_score <= 0.0:
            stay_penalty = RewardComponents.STAY_NO_CRITIC
            loop_penalty = 0.0  # Override loop penalty

        return stay_penalty, loop_penalty

    def _reset_eligibility_scores(self):
        """Reset eligibility scores after successful rebalancing."""
        for cell in self._cells.values():
            if cell.get_eligibility_score() > 0.3:
                cell.set_eligibility_score(cell.get_eligibility_score() - 0.2)

        self._env_logger.info(
            message=f" ---------> {self._truck.get_cell().get_id()} has been rebalanced. "
            f"All eligibility scores adjusted ###################################"
        )
        self._truck.get_cell().set_eligibility_score(1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Graph Update
    # ─────────────────────────────────────────────────────────────────────────

    def _flows_from_event_buffer(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        ORACLE: expected flows read from the realised future trips in the event buffer.
        Returns ({cell: peak net outflow}, {cell: arrivals after that peak}) for the next
        TIMESLOT_DURATION_SECONDS.
        """
        expected_departures: dict[int, int] = {}
        expected_max_departure: dict[int, int] = {}
        expected_after_arrival: dict[int, int] = {}

        for event in self._event_buffer:
            if event.time > self._env_time + EnvDefaults.TIMESLOT_DURATION_SECONDS:
                break
            cell = None
            cell_id = -1
            if event.is_departure():
                loc = event.get_trip().get_start_location()
                if loc.get_station_id() != 10000:
                    cell = loc.get_cell()
                    if cell is None:
                        raise ValueError(
                            f"Departure event with invalid cell for station {loc.get_station_id()}"
                        )
                    cell_id = cell.get_id()
                    expected_departures[cell_id] = (
                        expected_departures.get(cell_id, 0) + 1
                    )
            elif event.is_arrival():
                loc = event.get_trip().get_end_location()
                if loc.get_station_id() != 10000:
                    cell = loc.get_cell()
                    if cell is None:
                        raise ValueError(
                            f"Arrival event with invalid cell for station {loc.get_station_id()}"
                        )
                    cell_id = cell.get_id()
                    expected_departures[cell_id] = (
                        expected_departures.get(cell_id, 0) - 1
                    )
            if cell is not None:
                previous_max = expected_max_departure.get(cell_id, 0)
                expected_max_departure[cell_id] = max(
                    previous_max, expected_departures[cell_id]
                )
                expected_after_arrival[cell_id] = (
                    expected_max_departure[cell_id] - expected_departures[cell_id]
                )

        return expected_max_departure, expected_after_arrival

    def _flows_from_forecast(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        Same quantities as the oracle, computed on the sampled Poisson forecast: for each
        cell, running net outflow (departures - arrivals) over the window
        [env_time, env_time + slot duration]; its peak (>= 0) and the arrivals after it.
        """
        window_start = self._env_time
        window_end = window_start + EnvDefaults.TIMESLOT_DURATION_SECONDS
        expected_max_departure: dict[int, int] = {}
        expected_after_arrival: dict[int, int] = {}

        for i, cell_id in enumerate(self._cell_order):
            times = self._forecast_times[i]
            lo = np.searchsorted(times, window_start, side="left")
            hi = np.searchsorted(times, window_end, side="right")
            if hi <= lo:
                continue
            net = np.cumsum(self._forecast_deltas[i][lo:hi], dtype=np.int64)
            peak = max(0, int(net.max()))
            expected_max_departure[cell_id] = peak
            expected_after_arrival[cell_id] = peak - int(net[-1])

        return expected_max_departure, expected_after_arrival

    def _update_cells_metrics(self):
        """Update the cell subgraph with current regional metrics."""

        if self._critic_mode == "oracle":
            expected_max_departure, expected_after_arrival = self._flows_from_event_buffer()
        else:
            expected_max_departure, expected_after_arrival = self._flows_from_forecast()

        # ── Update each cell ──────────────────────────────────────────────────────
        self._global_critic_score = 0
        for cell_id, cell in self._cells.items():
            expected = expected_max_departure.get(cell_id, 0) + self._min_bikes_per_cell
            aft_arrivals = expected_after_arrival.get(cell_id, 0)
            cell.update_metrics(
                stations=self._stations, expected=expected, aft_arrivals=aft_arrivals
            )
            cell.set_metric(
                "truck_cell",
                1.0 if cell.get_id() == self._truck.get_cell().get_id() else 0.0,
            )

            # FOR TEST
            raw_critic = cell.get_critic_score()

            if self._use_binary_critic:
                # Binary classification: critic = 1 if “bad”, else keep raw score.
                # Use set_metric (not set_critic_score) so the previous-step value
                # snapshot in 'old_critic_score' is not overwritten a second time
                # within the same update. The binary critic threshold (> 0.0) is
                # looser than Cell's own (> 0.05), so flag/surplus are forced here
                # to match what set_critic_score(1.0) used to do.
                is_critical = raw_critic > 0.0
                if is_critical:
                    cell.set_metric("critic_score", 1.0)
                    cell.set_is_critical(True)
                    cell.set_metric("surplus_bikes", 0)
                    self._global_critic_score += 1.0
            else:
                self._global_critic_score += max(raw_critic, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _bike_repositioning(
        self,
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

        Returns:
            Dictionary mapping cell_id to number of bikes.
        """
        # -------------------------------------------------------------------
        # Step 1: Uniform base allocation, adjusted if fleet is too small
        # -------------------------------------------------------------------
        available = self._maximum_number_of_bikes - self._truck.get_load()
        base_bikes_per_cell = self._min_bikes_per_cell
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

    # def _adjust_depot_system_discrepancy(self):
    #     """
    #     Ensure the total bike count matches the maximum fleet size.
    #
    #     Bikes that exit the system are replaced from the outside_system_bikes pool.
    #     """
    #     depot_load = len(self._depot.bikes)
    #     system_load = len(self._system_bikes)
    #     truck_load = self._truck.get_load()
    #     total_bikes = depot_load + system_load + truck_load
    #
    #     if total_bikes < self._maximum_number_of_bikes:
    #         bikes_to_add = self._maximum_number_of_bikes - total_bikes
    #
    #         for _ in range(bikes_to_add):
    #             bike_id = next(iter(self._outside_system_bikes))
    #             bike = self._outside_system_bikes.pop(bike_id)
    #             bike.reset()
    #             self._depot.bikes[bike.get_bike_id()] = bike
    #
    #         self._logger.warning(
    #             message=f" ---------> System has been adjusted to max_number_of_bikes "
    #                     f"adding {bikes_to_add} bikes"
    #         )
