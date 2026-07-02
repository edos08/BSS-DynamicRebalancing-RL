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

# Create environment
env = gym.make(
    "gymnasium_env/FullyDynamicEnv-v0",
    data_path="data/",
    results_path="results/"
)

# Reset with configuration
obs, info = env.reset(options={
    'day': 'monday',
    'timeslot': 0,
    'total_timeslots': 56,
    'maximum_number_of_bikes': 3500,
    'depot_id': 1
})

# Training loop
done = False
while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### StaticEnv - TSP-Based Baseline

```python
import gymnasium as gym

# Create static environment for baseline comparison
env = gym.make(
    "gymnasium_env/StaticEnv-v0",
    data_path="data/"
)

obs, info = env.reset(options={
    'day': 'monday',
    'maximum_number_of_bikes': 3500,
    'num_rebalancing_events': 2  # Rebalancing operations per day
})
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

The observation is a graph-structured state containing:
- **Node features** (per cell): bike count, critic score, battery levels, demand/arrival rates, rebalancing history, truck presence
- **Edge features**: normalized distances between adjacent cells
- **Global features**: simulation time, truck load, depot inventory

#### Reward Function

Complex reward encouraging efficient rebalancing:
- **Base cost**: -0.1 per step (encourages efficiency)
- **Invalid actions**: -1.0 penalty
- **Self-loops**: -0.6 penalty (back-and-forth movements)
- **Drop rewards**: Up to +2.0 for rebalancing critical cells
- **Pick-up rewards**: Positive for removing bikes from surplus cells
- **Charge rewards**: Up to +0.5 for useful battery swaps
- **Stay penalties**: -1.0 in critical cells, +0.3 if no critical cells exist

#### Reset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `day` | str | `'monday'` | Starting day of week |
| `timeslot` | int | `0` | Starting timeslot (0-7, 3-hour intervals) |
| `total_timeslots` | int | `56` | Total simulation duration (7 days = 56 slots) |
| `maximum_number_of_bikes` | int | `500` | Total bike fleet size |
| `depot_id` | int | `1` | Cell ID of bike depot |

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
| `day` | str | `'monday'` | Starting day of week |
| `maximum_number_of_bikes` | int | — | Total bike fleet size |
| `num_rebalancing_events` | int | `1` | Rebalancing operations per day |
| `fixed_rebal_bikes_per_cell` | int | `5` | Base bikes per cell during rebalancing |
| `depot_id` | int | `491` | Cell ID of bike depot |

#### How It Works

1. Bikes distributed based on predicted net flow
2. At scheduled intervals (e.g., every 12 hours), system performs TSP-based rebalancing
3. Surplus bikes picked up and delivered to deficit cells
4. No continuous agent decisions; fully automated

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
│   ├── network.graphml       # OSM street network
│   ├── cell_data.pkl                   # Spatial grid cells
│   ├── distance_matrix.csv             # Station-to-station distances
│   ├── nearby_nodes.pkl                # Neighborhood lookup
│   ├── ev_velocity_matrix.csv          # Time-varying travel speeds
│   ├── ev_consumption_matrix.csv       # Expected trip volumes
│   └── global_rates.pkl                # Demand rates by timeslot
├── matrices/09-10/                     # PMF matrices (origin-destination probabilities)
└── rates/09-10/                        # Poisson rate matrices per timeslot
```

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

#### FullyDynamicEnv

```python
MAX_BIKES = 500                # Default fleet size
MAX_TRUCK_LOAD = 30            # Truck capacity
INITIAL_TRUCK_BIKES = 15       # Starting truck load
TIMESLOT_DURATION_HOURS = 3    # Episode divided into 3-hour slots
STEP_DURATION_SECONDS = 30     # Simulation step size
DISCOUNT_FACTOR = 0.99         # RL discount factor
```

#### StaticEnv

```python
DEFAULT_BIKES_PER_CELL = 5            # Base allocation per cell
DEFAULT_NUM_REBALANCING_EVENTS = 1    # Daily rebalancing operations
```

---

## Advanced Usage

### Custom Reward Function

Modify `FullyDynamicEnv._compute_reward()` to implement custom reward shaping:

```python
def _compute_reward(self, action, time, distance, invalid_action):
    # Your custom reward logic
    reward = 0.0
    # ... implementation ...
    return reward
```

### Graph Neural Network Integration

The environment provides a NetworkX graph representation suitable for GNNs:

```python
# Access the cell subgraph
subgraph = env._cell_subgraph

# Node features available:
# - truck_cell, low_battery_bikes, rebalanced
# - failure_rates, failures, bikes, critic_score
# - visits, operations, eligibility_score
```

### Logging and Debugging

Enable detailed logging:

```python
env._logger.set_logging(True)
```

Logs include:
- Action execution details
- Trip failures and bike availability
- Truck movements and load changes
- State transitions

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
cd preprocessing
bss-preprocess --data-path /path/to/data --steps download preprocess grid
```

### Memory Issues

**Issue**: Environment crashes with `MemoryError`

**Solution**: Reduce fleet size or simulation duration:
```python
env.reset(options={
    'maximum_number_of_bikes': 1000,    # Reduce from 3500
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
- Disable logging: `env._logger.set_logging(False)`

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

