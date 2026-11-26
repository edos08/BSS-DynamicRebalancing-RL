# Gymnasium Environment

Gymnasium-compatible reinforcement learning environments for bike-sharing system dynamic rebalancing.

This package provides two Gymnasium environments for simulating and optimizing bike rebalancing operations in a bike-sharing system. The environments model realistic bike trip demand, battery consumption, and rebalancing truck operations based on real-world data from the BlueBikes system in Cambridge, MA.

---

## Installation

Install in editable mode for development:

```bash
cd gymnasium_env
pip install -e .
```

For production use:

```bash
pip install .
```

---

## Quick Start

### FullyDynamicEnv - RL Agent Controlled Rebalancing

```python
import gymnasium as gym
import gymnasium_env  # registers the environments

env = gym.make(
    "gymnasium_env/FullyDynamicEnv-v0",
    data_path="data/",
    results_path="results/",
    seed=42,
)

obs, info = env.reset(options={
    "total_timeslots": 56,            # 7 days x 8 slots
    "maximum_number_of_bikes": 500,
    "minimum_number_of_bikes": 1,     # minimum bikes per cell at start
    "depot_id": 18,                   # must be a cell id of your grid
})

done = False
while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, done, timeslot_done, info = env.step(action)

env.close()
```

Note: `step()` returns `(observation, reward, done, timeslot_terminated, info)`.
The 4th value flags the end of a 3-hour timeslot (not a Gymnasium `truncated`); `done` marks the end of the episode.

The first `reset()` precomputes a whole week of trip events (it can take a while and is cached in `data/.cache`);
afterwards the next episode is computed in a background process while the current one runs.
Call `env.close()` to stop that process.

### StaticEnv - TSP-Based Baseline

```python
env = gym.make("gymnasium_env/StaticEnv-v0", data_path="data/", results_path="results/")

obs, info = env.reset(options={
    "total_timeslots": 56,
    "maximum_number_of_bikes": 500,
    "depot_id": 18,
    "num_rebalancing_events": 2,         # rebalancing operations per day
    "starting_rebalancing_event": 1,     # hour of the first one (1 -> 01:00 and 13:00)
})
done = False
while not done:
    _, _, done, _, info = env.step(0)    # the action is ignored
```

---

## Environments

### FullyDynamicEnv-v0

The fully dynamic environment where an RL agent controls a rebalancing truck to minimize service failures and optimize bike distribution across the network.

#### Action Space

Discrete action space with 8 actions:

| Action | Value | Description |
|--------|-------|-------------|
| `STAY` | 0 | Wait at current cell (60 seconds) |
| `UP` | 1 | Move truck to adjacent cell above |
| `DOWN` | 2 | Move truck to adjacent cell below |
| `LEFT` | 3 | Move truck to adjacent cell left |
| `RIGHT` | 4 | Move truck to adjacent cell right |
| `DROP_BIKE` | 5 | Drop a bike at current cell |
| `PICK_UP_BIKE` | 6 | Pick up a bike from current cell |
| `CHARGE_BIKE` | 7 | Swap low-battery bike with charged one |

#### Observation Space

A flat `Box` vector of size `17 + 2 × n_cells` (float16):

| Part | Size | Content |
|------|------|---------|
| scalars | 5 | system bike ratio, depot bike ratio, surplus flag of truck cell, empty flag of truck cell, global critic score / n_cells |
| previous action | 8 | one-hot of the last action |
| truck position | n_cells | one-hot of the truck cell |
| critical cells | n_cells | 1 if the cell is critical |
| borders | 4 | 1 for each missing neighbour (up, down, left, right) |

The spatial/graph information is exposed separately through `info["cell_dict"]` (a `{cell_id: Cell}` dict that is updated in place)
together with `info["nodes_dict"]` and `info["distance_lookup"]` (returned by `reset`). The trainer builds a graph of
cells from these (`rl_training.utils.build_cell_graph_from_cells`) to feed a GNN.
Cell features available through `Cell.get_all_metrics()`: `truck_cell`, `critic_score`, `eligibility_score`, `total_bikes`,
`surplus_bikes`, `dead_bikes`, `demand_rate`, `arrival_rate`, `failures`, `failure_rate`, `visits`, `operations`,
`pick_ups`, `drops`, `total_rebalanced`, ...

#### Critic score

For each cell, `critic = -1 + exp((1 + 0.2·bikes_arriving_later) · (1 − bikes / expected))`, where `expected` is the peak
net outflow (departures − arrivals) expected in the next 3 hours plus `minimum_number_of_bikes`, and
`bikes_arriving_later` the arrivals expected after that peak. A cell is *critical* when the score is
positive (with `use_binary_critic=True`, the default, critical cells have score exactly 1.0).
`surplus_bikes` counts the bikes a cell could give away without becoming critical.

How the next 3 hours are anticipated is set by `critic_mode`:

| `critic_mode` | Source of the expected flows |
|---------------|------------------------------|
| `"poisson"` (default) | A **forecast sampled from the demand model**, plus what is **already known**. The forecast draws whole trips the way the simulator does (Poisson number of trips from the slot's global rate, origin/destination from the PMF, travel time from distance and a truncated-Gaussian speed) with its own RNG, independently of the trips the simulator will really generate. Trips **already on the road** are taken from the simulation (their arrival times are known), and a forecast arrival counts only while its departure is still in the future, so nothing is counted twice. The agent has **no look-ahead** on trips that have not started. |
| `"oracle"` | Reads the **realised** future trips from the simulator's event buffer (the future is known exactly). Kept for ablations. |

The forecast is drawn once per timeslot (two slots ahead, like the event buffer) and then slides with the clock; a new
scenario is drawn only for each newly entered slot (about 2 ms per slot on the toy grid, vs ~190 steps per slot). It is deliberately *not* redrawn at every step: the reward contains the
change of the global critic score, and resampling would turn the forecast noise into reward noise. The forecast is reproducible
for a given `reset(seed=...)`. Note that it is a single noisy scenario: it is unbiased, but a cell's peak net outflow has a standard
deviation of the same order as its mean.

The oracle also leaks into the rewards and observation, so results obtained with `critic_mode="oracle"` are optimistic.

#### Reward Function

All constants live in `RewardComponents` (`fully_dynamic_env.py`). Main terms:

| Term | Value |
|------|-------|
| Base cost per step | -0.1 |
| Invalid action (returns immediately) | -1.0 |
| Back-and-forth loop (UP/DOWN, LEFT/RIGHT, STAY/STAY, PICK_UP/DROP) | -0.6 |
| Drop in a critical cell that becomes balanced | +2.0 |
| Drop in a critical cell that stays critical | +1.0 |
| Drop in a surplus cell | -0.5 |
| Drop elsewhere | +0.01 |
| Pick-up from a critical cell / that makes the cell critical | -0.5 / -2.0 |
| Pick-up from a surplus cell | +0.2 |
| Charge a low-battery bike (critical / normal cell) | +0.5 / +0.3 |
| Charge a bike that did not need it (critical / normal cell) | -0.1 / -0.3 |
| Move to a recently visited, non-critical cell | -0.2 (+0.1 × border type of the cell left) |
| Move to a barely visited cell | +0.3 |
| Move with an empty truck | -0.05 |
| Stay in a critical cell | -1.0 |
| Stay otherwise / when no cell is critical | -0.1 / 0.0 |
| Deployment / idle depot shaping | +0.02·deployed ratio / -0.02·depot ratio |
| Change of global critic score | -0.5 · Δ |
| Final step of the episode | -(total failures / total_timeslots) |

#### Reset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `day` | str | `'monday'` | Starting day of week |
| `timeslot` | int | `0` | Starting timeslot (0-7, 3-hour intervals) |
| `total_timeslots` | int | `56` | Total simulation duration (7 days = 56 slots) |
| `maximum_number_of_bikes` | int | `1000` | Total bike fleet size |
| `minimum_number_of_bikes` | int | `5` | Minimum bikes per cell at the start |
| `enable_repositioning` | bool | `True` | Distribute the bikes above the minimum (random, or by net flow) instead of leaving them in the depot |
| `use_net_flow` | bool | `False` | Bias that distribution by predicted net outflow |
| `use_binary_critic` | bool | `True` | Binary critical flag as critic score |
| `critic_mode` | str | `"poisson"` | `"poisson"` (sampled forecast, no look-ahead) or `"oracle"` (realised future trips) |
| `depot_id` | int | `1` | Cell id of the depot |
| `initial_cell` | int | random | Truck starting cell |
| `max_truck_load` | int | `30` | Truck capacity (starts with 15 bikes) |
| `discount_factor` | float | `0.99` | Stored for the agent |
| `results_path` | str | — | Per-episode log directory (enables env logging for that episode) |

`reset(seed=...)` reseeds the environment, the action space and the trip sampling.

#### Key Features

- **Event-driven simulation**: Realistic Poisson-distributed bike trip generation
- **Battery management**: Bikes consume battery; low-battery bikes cannot be rented
- **Spatial grid**: City divided into cells; truck navigates grid
- **Dynamic demand**: Time-varying demand based on real historical data
- **Eligibility tracking**: Cells track rebalancing history to encourage exploration

### StaticEnv-v0

Baseline environment using periodic TSP-based system-wide rebalancing (no continuous agent control).

#### Reset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `total_timeslots` | int | `56` | Total simulation duration |
| `maximum_number_of_bikes` | int | `1000` | Total bike fleet size |
| `minimum_number_of_bikes` | int | `5` | Minimum bikes per cell |
| `num_rebalancing_events` | int | `2` | Rebalancing operations per day |
| `starting_rebalancing_event` | int | `1` | Hour of the first rebalancing; the others follow every `24 // num_rebalancing_events` hours |
| `enable_repositioning` / `use_net_flow` | bool | `True` / `True` | Initial distribution (see FullyDynamicEnv) |
| `depot_id` | int | `1` | Cell id of the depot |

`step()` ignores the action and simulates **one whole 3-hour timeslot**. The returned observation is `{}` and the reward `0`.
`info["failures"]` is the number of failed trips in that timeslot (an `int`), and `info["rebalance_times"]` lists the
duration (s) of the rebalancing operations performed in it (`[0]` if none). `enable_repositioning` / `use_net_flow` only
affect the initial distribution of the bikes.

#### How It Works

1. Bikes are distributed at reset (base allocation, optionally biased by predicted net flow)
2. At each scheduled hour the target distribution (at least 1 bike per cell, extra bikes proportional to the predicted net
   outflow of the next window) is computed
3. A TSP route over the surplus / deficit cells gives the time a truck would need; this time is reported in `rebalance_times`
4. The redistribution itself is applied **instantaneously**: all bikes at stations and in the depot are pooled and
   re-placed according to the target (leftovers return to the depot)

---

## Package Structure

```
gymnasium_env/
├── README.md
├── pyproject.toml
└── src/
    └── gymnasium_env/
        ├── __init__.py                     # Gymnasium environment registration
        ├── envs/
        │   ├── __init__.py
        │   ├── fully_dynamic_env.py        # RL-controlled environment
        │   └── static_env.py               # TSP baseline environment
        └── simulator/
            ├── __init__.py
            ├── bike.py                     # Bike entity (battery, status)
            ├── bike_simulator.py           # Event-driven trip simulation
            ├── cell.py                     # Spatial grid cell
            ├── event.py                    # Event types (departure/arrival)
            ├── station.py                  # Station entity (bike docking)
            ├── trip.py                     # Trip entity (origin/destination)
            ├── truck.py                    # Rebalancing truck
            ├── truck_simulator.py          # Truck movement and operations
            └── utils.py                    # Helper functions and constants
```

---

## Data Requirements

The environments require preprocessed data in the specified `data_path`:

```
data/
├── utils/
│   ├── network.graphml                 # OSM street network
│   ├── cell_data.pkl                   # Spatial grid cells (dict of Cell objects)
│   ├── distance_matrix.csv             # Node-to-node distances (rows: node_id, columns: str(node_id))
│   ├── nearby_nodes.pkl                # Neighborhood lookup
│   ├── ev_velocity_matrix.csv          # Truck speed by hour (rows) and day (columns)
│   ├── ev_consumption_matrix.csv       # EV consumption by hour and day
│   └── global_rates.pkl                # {(day, timeslot): trips per second}
└── matrices/09-10/<day>/NN-pmf-matrix.csv   # Origin-destination PMF per day (monday...) and timeslot (00-07)
```

The matrices folder name (`matrices/09-10`) is fixed in `simulator/utils.py` (`EnvPaths.matrices_folder`); it matches the
default `months: [9, 10]` of the bundled sources. If you preprocess other months, update `EnvPaths` accordingly.
The virtual node `10000` (trips starting/ending outside the covered area) must be present in the PMF matrices.

Generate this data using the [`preprocessing`](../preprocessing) package.

---

## Simulator Components

### Core Entities

- **Bike**: Tracks battery level, location, and rental status
- **Station**: Docking point at network node; manages available bikes
- **Cell**: Spatial region containing multiple stations; truck navigation unit
- **Truck**: Rebalancing vehicle with capacity and location
- **Trip**: User trip from origin to destination
- **Event**: Discrete event (departure/arrival) in event-driven simulation

### Simulators

- **BikeSimulator**: Handles event processing (trip departures/arrivals), bike availability checks, battery validation, nearby station fallback
- **TruckSimulator**: Implements truck actions (movement, pickup, drop, charge), depot refilling, TSP-based rebalancing

### Utilities

- **Actions**: Enum for action types
- **Logger**: Detailed logging for debugging and analysis
- **Helper functions**: Distance calculation, Poisson event generation, graph initialization, battery sampling

---

## Configuration

### Environment Constants

`EnvDefaults` in `fully_dynamic_env.py` (the trainer overrides the fleet / repositioning ones through `reset(options=...)`):

```python
MAX_BIKES = 1000               # Default fleet size
MIN_BIKES_PER_CELL = 5
MAX_TRUCK_LOAD = 30            # Truck capacity
INITIAL_TRUCK_BIKES = 15       # Starting truck load
TIMESLOT_DURATION_HOURS = 3    # Episode divided into 3-hour slots (first slot starts at 01:00)
STEP_DURATION_SECONDS = 30     # Simulation tick
DISCOUNT_FACTOR = 0.99
ELIGIBILITY_DECAY = 0.9968     # Per tick; border cells use 0.99
```

`StaticEnv` has its own `EnvDefaults` (`NET_FLOW_BASED_REPOSITIONING = True`, `INITIAL_TRUCK_BIKES = 0`,
`DEFAULT_NUM_REBALANCING_EVENTS = 2`, `DEFAULT_STARTING_REBALANCING_EVENT = 1`).

---

## Advanced Usage

### Custom Reward Function

Edit the constants in `RewardComponents`, or the logic in `FullyDynamicEnv._get_reward()` and its `_compute_*` helpers:

```python
def _get_reward(self, action, old_eligibility_score, old_critic_score,
                old_global_critic_score, old_surplus_bikes) -> float:
    ...
```

### Graph Neural Network Integration

Build a graph of cells from the objects returned by `reset()` / `step()`:

```python
from rl_training import build_cell_graph_from_cells, update_cell_graph_features, convert_graph_to_data

graph = build_cell_graph_from_cells(info["cell_dict"], info["nodes_dict"], info["distance_lookup"])
# after each step:
update_cell_graph_features(graph, info["cell_dict"])
data = convert_graph_to_data(graph, node_features=["truck_cell", "critic_score", "eligibility_score", "total_bikes"])
```

### Logging and Debugging

Pass `logging_enabled=True` to `gym.make`, or give a `results_path` in `reset(options=...)` to log that episode
to `<results_path>/logs/env.log`. Logs include the executed actions, trip failures, truck movements and state transitions.

---

## Troubleshooting

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'gymnasium_env'`

**Solution**: Install in editable mode from package root:
```bash
cd gymnasium_env
pip install -e .
```

### Missing Data Files

**Issue**: `FileNotFoundError` when creating environment

**Solution**: Run preprocessing pipeline first:
```bash
bss-preprocess --data-path /path/to/data --source bluebikes
```

### Memory Issues

**Issue**: Environment crashes with `MemoryError`

**Solution**: Reduce fleet size or simulation duration:
```python
env.reset(options={
    'maximum_number_of_bikes': 300,
    'total_timeslots': 16               # Reduce from 56 (2 days instead of 7)
})
```

### Slow Simulation

**Issue**: Environment step() is slow

**Causes**:
- Large fleet size increases event processing overhead
- Dense spatial grid requires more truck movement computations
- Enable logging increases I/O overhead

**Solutions**:
- Reduce bike fleet size
- Increase cell size in preprocessing
- Keep logging disabled (the default)

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** (this package) → Uses preprocessed data for simulation
3. **RL Training** → Trains agent in simulated environment
4. **Benchmark** → Compares against baselines
5. **Results WebApp** → Visualizes training progress

---

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@mastersthesis{scarpel2025bss,
  title   = {Fully Dynamic Rebalancing of Dockless Bike Sharing Systems 
             using Deep Reinforcement Learning},
  author  = {Scarpel, Edoardo},
  year    = {2025},
  school  = {Università degli Studi di Padova},
  url     = {https://hdl.handle.net/20.500.12608/84368}
}
```

---

## License

See root LICENSE file for details.

---

## Contributing

Part of the BSS Dynamic Rebalancing RL monorepo.  
See main repository for contribution guidelines.

---

## Author

**Edoardo Scarpel**  
Ph.D. Student, University of Padova  
Email: edoardo.scarpel@phd.unipd.it

