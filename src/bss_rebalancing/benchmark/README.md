# Benchmark

Baseline evaluation framework for bike-sharing system rebalancing strategies using the static TSP-based environment.

This package provides benchmarking capabilities for evaluating static rebalancing strategies in bike-sharing systems. It runs simulations using the `StaticEnv` environment, which performs periodic system-wide rebalancing using Traveling Salesman Problem (TSP) optimization.

The benchmark serves as a baseline for comparing against reinforcement learning approaches by measuring system performance metrics under deterministic rebalancing schedules.

---

## Installation

Install in editable mode for development:

```bash
cd benchmark
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/benchmark
```

---

## Quick Start

### Basic Usage

Run a benchmark with default settings:

```bash
bss-benchmark --data-path data/ --results-path results/
```

### Custom Configuration

```bash
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --run-id 0 \
    --num-seed-runs 5 \
    --max-num-bikes 300 \
    --num-rebal-events 2 \
    --start-rebal-hour 1
```

### Python API

```python
from benchmark import run_benchmark

run_benchmark({
    'run_id': 0,
    'data_path': 'data/',
    'results_path': 'results/',
    'seed': 42,
    'num_seed_runs': 1,
    'num_rebal_events': 2,
    'start_rebal_hour': 1,
    'maximum_number_of_bikes': 300,
    'minimum_number_of_bikes': 1,
    'enable_repositioning': False,
    'use_net_flow': False,
    'log': False,
})
```

---

## Package Structure

```
benchmark/
├── README.md
├── pyproject.toml
└── src/
    └── benchmark/
        ├── __init__.py          # Package initialization and exports
        ├── run.py               # Main benchmark runner (CLI entry point)
        ├── results_manager.py   # EpisodeResults / ResultsManager (same layout as rl_training)
        ├── logging_config.py    # Logging setup
        └── utils.py             # Cell-graph helpers, seeding
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | int | `0` | Unique identifier for this benchmark run |
| `--data-path` | str | `"data/"` | Path to preprocessed data directory |
| `--results-path` | str | `"results/"` | Root results directory (a `run_NNN/benchmark/` folder is created) |
| `--max-num-bikes` | int | `500` | Total bike fleet size |
| `--min-num-bikes` | int | `1` | Minimum bikes per cell |
| `--num-rebal-events` | int | `2` | Rebalancing events per day |
| `--start-rebal-hour` | int | `1` | Clock hour of the first rebalancing event (the others follow every `24 // num-rebal-events` hours) |
| `--enable-repositioning` | flag | — | Distribute the bikes above the per-cell minimum at episode start |
| `--use-net-flow` | flag | — | Bias that distribution by predicted net flow |
| `--seed` | int | `42` | Random seed |
| `--num-seed-runs` | int | `1` | Number of weekly episodes (seed, seed+1, ...) |
| `--log` | flag | — | Enable detailed environment logging |

The episode length is fixed to 56 timeslots (one week); the depot cell is `18` (`params["depot_position_id"]` in `run.py`).

---

## Configuration

Default `params` in `run.py`:

```python
params = {
    "seed": 42,
    "num_episodes": 1,
    "total_timeslots": 56,              # 1 week (8 timeslots/day x 7 days)
    "num_rebalancing_events": 2,        # Rebalancing events per day
    "starting_rebalancing_event": 1,    # Clock hour of the first event
    "enable_repositioning": False,
    "use_net_flow": False,
    "maximum_number_of_bikes": 500,
    "minimum_number_of_bikes": 1,
    "depot_position_id": 18,            # Depot cell id
    "initial_cell_id": 18,
}
```

---

## How It Works

### Static Environment

The benchmark uses the `StaticEnv` environment, which:

1. **Distributes bikes** at the start of the episode (minimum per cell, optionally biased by predicted net flow)
2. **Simulates trips** using the same Poisson trip generator as the RL environment
3. **Rebalances at scheduled hours** (`num-rebal-events` per day, the first at `start-rebal-hour`)
4. **Uses TSP** over the surplus / deficit cells to compute how long the rebalancing takes
5. **Records metrics**, including failures and rebalancing time

### Rebalancing Strategy

At each scheduled hour:
- A target distribution is computed (at least 1 bike per cell, the rest proportional to the predicted net outflow)
- Surplus and deficit cells are identified; a TSP route over them gives the rebalancing duration
- The redistribution is applied instantaneously (all bikes at stations and in the depot are re-placed), so the duration is
  reported but not simulated as truck unavailability

### Metrics Collected

**Per timeslot** (`timeslot_metrics.csv`): failures, demand, deployed / depot / truck / outside / traveling bikes,
rebalancing durations (s), global critic score

**Per episode** (`scalars.json`, `bench_summary.csv`): `total_failures`, `mean_daily_failures`

**Spatial** (`cell_subgraph.gpickle`): per-cell failures, failure rate, mean critic score, mean bikes

---

## Results Output

Results use the same structure as the training runs, so the results web app can display them next to the RL results:

```
results/
└── run_000/
    └── benchmark/
        ├── config.json              # Parameters used
        ├── bench_summary.csv        # One row per episode
        ├── logs/
        └── episode_000/
            ├── scalars.json
            ├── timeslot_metrics.csv
            └── cell_subgraph.gpickle
```

### Loading Results

```python
import json
import pandas as pd

run = "results/run_000/benchmark"
summary = pd.read_csv(f"{run}/bench_summary.csv")
print(summary[["episode", "total_failures", "mean_daily_failures"]])

metrics = pd.read_csv(f"{run}/episode_000/timeslot_metrics.csv")
print(metrics[["timeslot", "failures", "demand"]].head())
```

---

## Usage Examples

### Multi-Episode Benchmark

Five weekly episodes with seeds 42..46:

```bash
bss-benchmark --data-path data/ --results-path results/ --run-id 2 --num-seed-runs 5
```

### Fleet Size Comparison

```bash
bss-benchmark --run-id 10 --max-num-bikes 200
bss-benchmark --run-id 11 --max-num-bikes 300
bss-benchmark --run-id 12 --max-num-bikes 500
```

### Rebalancing Frequency Study

```bash
bss-benchmark --run-id 20 --num-rebal-events 1
bss-benchmark --run-id 21 --num-rebal-events 2
bss-benchmark --run-id 22 --num-rebal-events 4 --start-rebal-hour 6
```

---

## Comparison with RL Training

This benchmark provides a baseline for comparing against RL approaches:

| Metric | Static Benchmark | RL Training |
|--------|------------------|-------------|
| **Decision Making** | Scheduled, deterministic | Learned, adaptive |
| **Rebalancing** | Periodic TSP-based | Continuous agent control |
| **Flexibility** | Fixed schedule | Responds to demand patterns |
| **Computation** | TSP at intervals | Neural network inference per step |

---

## Troubleshooting

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'benchmark'`

**Solution**: Install in editable mode:
```bash
cd benchmark
pip install -e .
```

### Missing Data Files

**Issue**: `FileNotFoundError` when running benchmark

**Solution**: Ensure preprocessing pipeline has been run:
```bash
cd preprocessing
bss-preprocess --data-path /path/to/data --source bluebikes
```

### Environment Not Found

**Issue**: `gymnasium.error.UnregisteredEnv: StaticEnv-v0 not found`

**Solution**: Install gymnasium_env package:
```bash
cd gymnasium_env
pip install -e .
```

### Slow Execution

**Issue**: Benchmark runs very slowly

**Causes**:
- Large fleet size
- Many episodes/timeslots
- Dense spatial grid

**Solutions**:
- Reduce fleet size: `--max-num-bikes 200`
- Reduce the number of episodes: `--num-seed-runs 1`

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
3. **RL Training** → Trains agent in simulated environment
4. **Benchmark** (this package) → Compares against baselines
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
