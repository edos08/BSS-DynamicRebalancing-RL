import json
import math
import multiprocessing
import os
import pickle
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import polars as pl
from scipy.stats import truncnorm
from tqdm import tqdm

from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.station import Station

# ==============================================================================
# Configuration Dataclass
# ==============================================================================


@dataclass(frozen=True)
class EnvPaths:
    """
    Configuration for environment file paths.

    All paths are relative to the data_path provided to the environment.
    """

    graph_file: str = "utils/network.graphml"
    cell_file: str = "utils/cell_data.pkl"
    nearby_nodes_file: str = "utils/nearby_nodes.pkl"
    global_rates_file: str = "utils/global_rates.pkl"
    distance_matrix_file: str = "utils/distance_matrix.csv"
    velocity_matrix_file: str = "utils/ev_velocity_matrix.csv"
    consumption_matrix_file: str = "utils/ev_consumption_matrix.csv"
    matrices_folder: str = "matrices/09-10"
    rates_folder: str = "rates/09-10"
    trips_folder: str = "trips/"


# Default paths instance
DEFAULT_PATHS = EnvPaths()

# ==============================================================================
# Day/Time Mappings
# ==============================================================================

DAYS_TO_NUM: Dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

NUM_TO_DAYS: Dict[int, str] = {v: k for k, v in DAYS_TO_NUM.items()}

# ==============================================================================
# Action Mappings
# ==============================================================================


class Actions(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7


ACTION_TO_STR: Dict[int, str] = {
    Actions.STAY.value: "STAY",
    Actions.UP.value: "UP",
    Actions.DOWN.value: "DOWN",
    Actions.LEFT.value: "LEFT",
    Actions.RIGHT.value: "RIGHT",
    Actions.DROP_BIKE.value: "DROP",
    Actions.PICK_UP_BIKE.value: "PICK_UP",
    Actions.CHARGE_BIKE.value: "CHARGE",
}

STR_TO_ACTION: Dict[str, int] = {v: k for k, v in ACTION_TO_STR.items()}

# ==============================================================================
# Self-Loop Detection
# ==============================================================================

# Define valid 2-step back-and-forth patterns that constitute self-loops
SELF_LOOP_PATTERNS: FrozenSet[Tuple[int, int]] = frozenset(
    [
        (Actions.UP.value, Actions.DOWN.value),
        (Actions.DOWN.value, Actions.UP.value),
        (Actions.LEFT.value, Actions.RIGHT.value),
        (Actions.RIGHT.value, Actions.LEFT.value),
        (Actions.STAY.value, Actions.STAY.value),
        (Actions.PICK_UP_BIKE.value, Actions.DROP_BIKE.value),
        (Actions.DROP_BIKE.value, Actions.PICK_UP_BIKE.value),
    ]
)


def detect_self_loops(actions: Tuple[int, int]) -> bool:
    """
    Detect if a pair of consecutive actions forms a self-loop pattern.

    A self-loop is when the agent performs two actions that cancel each other out,
    such as moving up then down, or picking up then dropping a bike.

    Parameters:
        actions: A tuple of two consecutive action values

    Returns:
        True if the action pair is a self-loop pattern, False otherwise
    """
    return actions in SELF_LOOP_PATTERNS


# ==============================================================================

# ----------------------------------------------------------------------------------------------------------------------


def logistic_penalty_function(m=1, k=1, b=1, x=0):
    return m / (1 + math.exp(k * (b - x)))


def generate_poisson_events(
    rate, time_duration, rng: np.random.Generator | None = None
) -> list[int]:
    """
    Generate Poisson events within a specified time duration.

    Parameters:
        - rate (float): The average rate of events per unit time.
        - time_duration (float): The total time duration in which events can occur.
        - rng: Optional numpy Generator for thread-safe reproducible sampling.
               If None, uses np.random.exponential (original behaviour).

    Returns:
        - list: A list of event times occurring within the specified time duration.
    """
    n_samples = int(rate * time_duration) + 100
    if rng is not None:
        inter_arrival_times = rng.exponential(1 / rate, n_samples)
    else:
        inter_arrival_times = np.random.exponential(1 / rate, n_samples)

    event_times = np.cumsum(inter_arrival_times)
    event_times = event_times[event_times < time_duration]

    return np.floor(event_times).astype(int).tolist()


def convert_seconds_to_hours_minutes_day(seconds, day=None) -> str:
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24
    minutes, seconds = divmod(remainder, 60)
    if day is not None:
        return f"{day} AT {hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def truncated_gaussian(
    lower=5, upper=25, mean=15, std_dev=5, rng: np.random.Generator | None = None
):
    """
    Samples one point from a truncated gaussian.

    Parameters:
        - lower (int): The lower bound of the truncation.
        - upper (int): The upper bound of the truncation.
        - mean (int): The mean value of the gaussian.
        - std_dev (int): The standard deviation of the gaussian.
        - rng: Optional numpy Generator for thread-safe reproducible sampling.
               If None, falls back to scipy's default (original behaviour).

    Returns:
        - speed (int): The sampled value from the truncated gaussian
    """
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    if rng is not None:
        # scipy truncnorm accepts a numpy Generator via the `random_state` parameter
        truncated_normal = truncnorm(a, b, loc=mean, scale=std_dev)
        speed = truncated_normal.rvs(random_state=rng)
    else:
        truncated_normal = truncnorm(a, b, loc=mean, scale=std_dev)
        speed = truncated_normal.rvs()
    return speed


# ----------------------------------------------------------------------------------------------------------------------


def initialize_graph(graph_path: str = "") -> nx.MultiDiGraph:
    """
    Initialize a road graph from a saved file.

    Parameters:
        - graph_path (str):  the file directory

    Returns:
        - nx.Graph: The graph representing the road network.
    """
    if os.path.isfile(graph_path):
        graph = ox.load_graphml(graph_path)
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError(
            "Network file does not exist. Please check the file path."
        )

    return graph


def initialize_bikes(station: Station | None = None, n: int = 0) -> dict[int, Bike]:
    """Initialize bikes at a station with auto-incrementing IDs."""
    bikes = {}
    for i in range(n):
        bike = Bike(station=station)
        bikes[bike.get_bike_id()] = bike
        if station is not None:
            station.lock_bike(bike)
    return bikes


def initialize_stations(
    stations: dict, depot_bikes: dict, bikes_per_station: dict
) -> tuple[dict, dict]:
    """
    Initialize a list of stations based on the nodes of the graph.

    Parameters:
        - stations (dict): Dictionary containing all the stations of the network plus the outside station with station.id=10000 .
        - depot (dict): Dictionary of all initialized bikes of the system.
        - bikes_per_station (dict): Dictionary that specifies ho many bikes are to be assigned to each station.

    Returns:
        - dict: A dictionary containing the bikes in the network, each assigned to a station.
        - dict: A dictionary containing the bikes outside the network, with bike.station=10000.
    """
    system_bikes = {}

    for station in stations.values():
        station_id = station.get_station_id()
        if station_id != 10000:
            total_bikes_for_station = bikes_per_station.get(station_id)
            bikes = {
                key: depot_bikes.pop(key)
                for key in list(depot_bikes.keys())[:total_bikes_for_station]
            }
            station.set_bikes(bikes)
            system_bikes.update(bikes)

    outside_system_bikes = initialize_bikes(n=1000)
    for bike in outside_system_bikes.values():
        bike.set_station(stations[10000])

    return system_bikes, outside_system_bikes


# ----------------------------------------------------------------------------------------------------------------------


def load_preprocessed_data(
    paths: EnvPaths, data_path: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[int, list[int]], dict, dict]:
    """
    Load preprocessed data for the environment.

    Parameters:
        - paths (EnvPaths): The paths configuration for the environment.
        - data_path (str): The base path where the preprocessed data is stored.

    Returns:
        - distance_matrix (pl.DataFrame): A DataFrame containing distances between nodes.
        - consumption_matrix (pl.DataFrame): A DataFrame containing energy consumption values between nodes.
        - velocity_matrix (pl.DataFrame): A DataFrame containing average velocity values between nodes.
        - nearby_nodes_dict (dict[int, list[int]]): A dictionary mapping each node to
            a list of nearby nodes within a specified radius.
        - global_rates (dict): A dictionary containing global demand and arrival rates for each node.
        - cell_dict (dict[int, Cell]): A dictionary mapping cell IDs to Cell objects.
    """

    distance_matrix_path = os.path.join(data_path, paths.distance_matrix_file)
    velocity_matrix_path = os.path.join(data_path, paths.velocity_matrix_file)
    consumption_matrix_path = os.path.join(data_path, paths.consumption_matrix_file)
    nearby_nodes_path = os.path.join(data_path, paths.nearby_nodes_file)
    global_rates_path = os.path.join(data_path, paths.global_rates_file)
    cell_data_path = os.path.join(data_path, paths.cell_file)

    distance_matrix = pl.read_csv(distance_matrix_path)
    consumption_matrix = pl.read_csv(consumption_matrix_path)
    velocity_matrix = pl.read_csv(velocity_matrix_path)

    with open(nearby_nodes_path, "rb") as file:
        nearby_nodes_dict = pickle.load(file)

    with open(global_rates_path, "rb") as file:
        global_rates = pickle.load(file)

    with open(cell_data_path, "rb") as file:
        cell_dict = pickle.load(file)

    return (
        distance_matrix,
        velocity_matrix,
        consumption_matrix,
        nearby_nodes_dict,
        global_rates,
        cell_dict,
    )


def flatten_pmf_matrix(
    pmf_matrix: pl.DataFrame,
) -> (
    pl.DataFrame
):  # TODO: precompute directly flattened PMF matrices to avoid this step at runtime
    """Flatten a PMF matrix for event simulation."""
    id_col = pmf_matrix.columns[0]  # First column (e.g., 'node_id')
    value_cols = pmf_matrix.columns[1:]  # All other columns

    flattened_pmf = (
        pmf_matrix.unpivot(
            index=id_col, on=value_cols, variable_name="col_id", value_name="value"
        )
        .with_columns(
            [
                # Create tuple ID: (row_id, col_id)
                pl.struct([pl.col(id_col), pl.col("col_id").cast(pl.Int64)]).alias(
                    "id"
                ),
                # Compute cumulative sum
                pl.col("value").cum_sum().alias("cumsum"),
            ]
        )
        .select(["id", "value", "cumsum"])
    )

    return flattened_pmf


# ==============================================================================


def cache_episode_zero(
    buffers: dict[int, list],
    cache_dir: str,
    seed: int,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    # Write buffer
    buffer_file = os.path.join(cache_dir, "episode_0.pkl")
    with open(buffer_file, "wb") as f:
        pickle.dump(buffers, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write metadata
    meta_file = os.path.join(cache_dir, "episode_0_meta.json")
    meta = {
        "seed": seed,
        "num_timeslots": len(buffers),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)


def load_episode_zero(
    cache_dir: str,
    seed: int,
    expected_timeslots: int,
) -> dict[int, list] | None:
    """
    Returns the cached buffer dict if the cache exists and the seed matches,
    otherwise returns None (caller must recompute).
    """
    buffer_file = os.path.join(cache_dir, "episode_0.pkl")
    meta_file = os.path.join(cache_dir, "episode_0_meta.json")

    if not os.path.exists(buffer_file) or not os.path.exists(meta_file):
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    if meta.get("seed") != seed or meta.get("num_timeslots") != expected_timeslots:
        print(
            f"[cache] Episode 0 cache mismatch "
            f"(cached seed={meta.get('seed')}, want seed={seed}) — recomputing."
        )
        return None

    with open(buffer_file, "rb") as f:
        buffers = pickle.load(f)
    return buffers


# ==============================================================================
# Episode Precomputation (shared between envs)
# ==============================================================================

def compute_buffer_logic(
        seed: int,
        data_path: str,
        global_rate_dict: dict,
        distance_lookup: dict,
        precomputed_episode_timeslots: int,
        timeslots_per_day: int,
        timeslot_duration_seconds: int,
        default_day: str,
        first_episode: bool = False,
        cache_dir: str | None = None,
        show_pbar: bool = False,
) -> dict:
    """
    Pure function — compute a full episode buffer of presampled trip events.
    No queue, no process state; safe to call from a subprocess.

    Parameters:
        seed: RNG seed for this episode.
        data_path: Base path where PMF matrix CSVs live.
        global_rate_dict: Maps (day_str, timeslot_int) → global arrival rate.
        distance_lookup: Node-keyed distance lookup dict.
        precomputed_episode_timeslots: Total number of timeslots to precompute
            (typically 7 days × timeslots_per_day).
        timeslots_per_day: Number of timeslots in one day.
        timeslot_duration_seconds: Duration of each timeslot in seconds.
        default_day: Name of the starting day (e.g. 'monday').
        first_episode: If True, attempt to load from / save to cache_dir.
        cache_dir: Directory for episode-zero caching. Ignored when None.
        show_pbar: Whether to render a tqdm progress bar.

    Returns:
        Dict mapping slot_index (int) → list of TripSample objects.
    """
    from gymnasium_env.simulator.bike_simulator import simulate_events

    # ── Cache load (first episode only) ─────────────────────────────────────
    if first_episode and cache_dir is not None:
        cached = load_episode_zero(
            cache_dir=cache_dir,
            seed=seed,
            expected_timeslots=precomputed_episode_timeslots,
        )
        if cached is not None:
            return cached

    # ── Compute ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    day = default_day
    buffers = {}
    pbar = (
        tqdm(
            total=precomputed_episode_timeslots,
            desc="[ENV] Computing episode",
            unit="slot",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        if show_pbar
        else None
    )

    for slot_index in range(precomputed_episode_timeslots):
        timeslot = slot_index % timeslots_per_day
        matrix_path = os.path.join(data_path, DEFAULT_PATHS.matrices_folder)
        pmf_matrix = pl.read_csv(
            os.path.join(
                matrix_path, day.lower(), str(timeslot).zfill(2) + "-pmf-matrix.csv"
            )
        )
        global_rate = global_rate_dict[(day.lower(), timeslot)]
        flattened_pmf = flatten_pmf_matrix(pmf_matrix)
        slot_seed = int(rng.integers(0, 2 ** 31))
        buffers[slot_index] = simulate_events(
            duration=timeslot_duration_seconds,
            timeslot=timeslot,
            global_rate=global_rate,
            pmf=flattened_pmf,
            distance_lookup=distance_lookup,
            seed=slot_seed,
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'seed': seed})
        if timeslot == timeslots_per_day - 1:
            day = NUM_TO_DAYS[(DAYS_TO_NUM[day] + 1) % 7]

    if pbar is not None:
        pbar.close()

    # ── Cache save (first episode only) ─────────────────────────────────────
    if first_episode and cache_dir is not None:
        cache_episode_zero(buffers, cache_dir=cache_dir, seed=seed)

    return buffers


def episode_worker(
        seed: int,
        data_path: str,
        global_rate_dict: dict,
        distance_lookup: dict,
        precomputed_episode_timeslots: int,
        timeslots_per_day: int,
        timeslot_duration_seconds: int,
        default_day: str,
        result_queue: multiprocessing.Queue,
        show_pbar: bool = False,
) -> None:
    """
    Thin subprocess entry point — calls compute_buffer_logic and puts the
    result in result_queue. Writes a crash log on failure and re-raises.

    Parameters mirror compute_buffer_logic; result_queue receives the
    returned buffer dict when computation succeeds.
    """
    try:
        result = compute_buffer_logic(
            seed=seed,
            data_path=data_path,
            global_rate_dict=global_rate_dict,
            distance_lookup=distance_lookup,
            precomputed_episode_timeslots=precomputed_episode_timeslots,
            timeslots_per_day=timeslots_per_day,
            timeslot_duration_seconds=timeslot_duration_seconds,
            default_day=default_day,
            show_pbar=show_pbar,
        )
        result_queue.put(result)
    except Exception as exc:
        # Write crash log so there's a file to inspect
        crash_log = os.path.join(data_path, "tmp", "bg_worker_crash.log")
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        tb = traceback.format_exc()
        with open(crash_log, "w") as f:
            f.write(tb)
        # Also push the exception into the queue so _acquire_next_episode_buffer
        # can surface it in the main process instead of silently timing out
        try:
            result_queue.put(exc)
        except Exception:
            pass  # queue itself is broken — crash log is the only record
        raise
