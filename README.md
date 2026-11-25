# 🚲 BSS Dynamic Rebalancing with Deep Reinforcement Learning

### Last-mile mobility. Real-time simulation. Adaptive rebalancing.

This repository accompanies the thesis project *Fully Dynamic Rebalancing of Dockless Bike Sharing Systems using Deep Reinforcement Learning*. It presents a novel framework for dynamically rebalancing bikes in a dockless **Bike Sharing System (BSS)** using a **Double Deep Q-Network (DDQN)** trained in a realistic, event-driven simulation environment.

---

## 🧠 Project Overview

Cities are increasingly adopting sustainable transport solutions to address congestion, emissions, and urban sprawl. Dockless BSSs offer flexible, green last-mile mobility—but their very flexibility introduces operational complexity. Bikes often cluster in popular zones, leaving others underserved.

This thesis tackles the challenge with a **fully dynamic rebalancing framework** driven by **Reinforcement Learning**, where decisions are made in real time based on real-world demand patterns and traffic conditions.

### 🎯 Highlights
- 🧠 A **DDQN agent** learns to make rebalancing decisions under uncertainty
- 🧪 **Event-driven simulation** using real **Cambridge, MA** demand data and **TomTom traffic profiles**
- 🛠️ Modular architecture with separate packages for preprocessing, training, validation, and benchmarking
- 📊 Real-time visualization dashboard for monitoring training progress
- 🔧 CLI tools for streamlined workflows

---

## 📄 Thesis

You can read the full thesis at [https://hdl.handle.net/20.500.12608/84368](https://hdl.handle.net/20.500.12608/84368)

---

## 📁 Project Structure

This is a **monorepo** containing five independent Python packages:

```
bss-rebalancing/
├── README.md                          # This file
├── pyproject.toml                     # Root build configuration
├── LICENSE                            # MIT License
└── src/
    └── bss_rebalancing/
        ├── preprocessing/             # Data preprocessing pipeline
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/preprocessing/
        │       ├── cli.py             # CLI entry point (bss-preprocess)
        │       ├── core/              # Graph, grid, plotting, source registry (sources.json), converter, manifest
        │       └── steps/             # Individual preprocessing steps
        │
        ├── gymnasium_env/             # Custom Gymnasium environment
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/gymnasium_env/
        │       ├── envs/              # Environment implementations
        │       └── simulator/         # Event-driven simulation engine
        │
        ├── rl_training/               # RL agent training and validation
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/rl_training/
        │       ├── agents/            # DQN agent implementation
        │       ├── networks/          # Neural Network architectures
        │       ├── memory/            # Replay buffer
        │       ├── train.py           # Training script (bss-train)
        │       └── validate.py        # Validation script (bss-validate)
        │
        ├── benchmark/                 # Baseline comparisons
        │   ├── README.md
        │   ├── pyproject.toml
        │   └── src/benchmark/
        │       └── run.py             # Benchmark runner (bss-benchmark)
        │
        └── results_webapp/            # Training visualization dashboard
            ├── README.md
            ├── pyproject.toml
            └── src/results_webapp/
                ├── app.py             # Dash application (bss-results-webapp)
                ├── callbacks.py       # Interactive callbacks
                ├── data_loader.py     # Results data loader
                └── plotting.py        # Plotting utilities
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/edos08/BSS-DynamicRebalancing-RL
   cd BSS-DynamicRebalancing-RL
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the project** (automatically installs all subpackages):
   ```bash
   pip install -e .
   ```

   This single command installs all five packages and their CLI tools in development mode.

4. **Verify installation**:
   ```bash
   bss-preprocess --help
   bss-train --help
   bss-validate --help
   bss-benchmark --help
   bss-results-webapp --help
   ```

---

## 📂 Data Sources

- 🚲 **BlueBikes trip data** (Cambridge, MA) - Real demand patterns
- 🛣️ **TomTom traffic speed profiles** - Time-dependent travel times
- 🌐 **OpenStreetMap** via osmnx - Street network topology

*(Note: Datasets are not included due to licensing restrictions. Download is automatic and instructions are provided in the preprocessing module.)*

---

## 🔧 Usage Workflow

### 1. Preprocess Data

Download and preprocess trip data, create spatial grid, and compute distance matrices:

```bash
# Full pipeline (the data source is mandatory; see core/sources.json)
bss-preprocess --data-path data/ --source bluebikes

# Specific steps only
bss-preprocess --data-path data/ --source bluebikes --steps download,preprocess,grid

# Skip certain steps
bss-preprocess --data-path data/ --source bluebikes --skip download

# Visualize the grid
bss-preprocess --data-path data/ --source bluebikes --plot grid-numbered
```

**Key arguments**:
- `--source`: Source id from `sources.json` (**required**; bundled: `bluebikes`, `citibike`)
- `--sources-json`: Custom sources file (default: the one bundled with the package)
- `--data-path`: Data directory (default: `data/`)
- `--steps` / `--skip`: Run / skip specific steps (`download, preprocess, interpolate, grid, distance, nodes, ev_matrices`)
- `--plot`: Visualization mode (`graph`, `grid`, `grid-numbered`)

Year, months, bounding box, place name and cell size are no longer CLI flags: they are defined per source in `sources.json`.
Adding a new bike-share feed means adding an entry there, not changing code.

See `bss-preprocess --help` for all options.

### 2. Train the RL Agent

Train a DDQN agent with the fully dynamic environment:

```bash
# Basic training
bss-train --data-path data/ --results-path results/

# Advanced configuration
bss-train \
    --data-path data/ \
    --results-path results/ \
    --run-id 1 \
    --num-episodes 150 \
    --max-num-bikes 400 \
    --exploration-time 0.7 \
    --device cuda:0 \
    --seed 42
```

**Key arguments**:
- `--run-id`: Experiment identifier (default: 0)
- `--num-episodes`: Training episodes (default: 140)
- `--max-num-bikes`: Fleet size (default: 500)
- `--min-num-bikes`: Minimum bikes per cell at episode start (default: 1)
- `--exploration-time`: Fraction of training over which epsilon decays to ~0.1 (default: 0.6)
- `--device` / `--val-device`: Hardware device for training / validation subprocesses (`cpu`, `cuda:0`, `mps`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--enable-repositioning`, `--use-net-flow`: Initial bike distribution strategy
- `--critic-mode`: `poisson` (default; the critic score uses a forecast sampled from the demand rates, no look-ahead on the
  simulated trips) or `oracle` (it reads the realised future trips; optimistic, useful for ablations)
- `--log`: Enable detailed environment logging
- `--one-validation`: Skip the in-training validations and validate only the final model

During training, once epsilon falls below 0.1 each episode snapshot is validated in a background
subprocess; the snapshot with the lowest `mean_daily_failures` is promoted to `models/best/`.

### 3. Validate the Model

Test a trained model on validation data:

```bash
# Validate the best model of a run
bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type best

# Validate a specific episode snapshot, or an arbitrary checkpoint
bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type episode --model-episode 42
bss-validate --run-id 0 --data-path data/ --results-path results/ \
    --model-path results/run_000/models/best/trained_agent.pt
```

**Key arguments**:
- `--run-id`: Run whose models/results are used (default: 0)
- `--model-type`: `best` (default), `episode` or `final`; with `episode` also pass `--model-episode`
- `--model-path`: Direct path to a `trained_agent.pt` (takes priority over `--model-type`)
- `--max-num-bikes` / `--min-num-bikes`: Fleet size / minimum bikes per cell
- `--total-timeslots`: Episode length (default: 56 = 1 week)
- `--num-seed-runs`: Number of validation episodes with incremented seeds (default: 1)

Validation uses a fixed epsilon of 0.01.

### 4. Run Benchmarks

Compare against baseline strategies:

```bash
# Basic benchmark
bss-benchmark --data-path data/

# With custom parameters
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --num-seed-runs 5 \
    --max-num-bikes 400 \
    --num-rebal-events 2
```

**Key arguments**:
- `--data-path`: Path to data folder (default: `data/`)
- `--results-path`: Where to save results (default: `results/`)
- `--num-seed-runs`: Number of weekly episodes with incremented seeds (default: 1)
- `--max-num-bikes` / `--min-num-bikes`: Fleet size (default: 500) / minimum bikes per cell
- `--num-rebal-events`: TSP rebalancing events per day (default: 2)

### 5. Visualize Training Progress

Launch the interactive dashboard to monitor training in real-time:

```bash
bss-results-webapp --results-path results/ --data-path data/ --port 8050
```

`--data-path` is optional; it is only needed to draw the spatial heatmaps (it must contain `utils/network.graphml`).
Then open http://localhost:8050 in your browser.

**Features**:
- 📊 Real-time metrics (failures, rewards, epsilon decay)
- 📈 Episode-level detailed analysis
- 🔍 Training dynamics (loss, Q-values, critic scores)
- 🗺️ Spatial failure patterns
- 🔄 Auto-refresh during training

---

## 📦 Package Details

### `preprocessing`
Handles data download, cleaning, interpolation, grid creation, and distance matrix computation.
- **CLI**: `bss-preprocess` (`--source` selects a feed from `sources.json`)
- **Key modules**: `download_trips`, `preprocess_data`, `preprocess_truck_grid`, `core/sources`, `core/converter`

### `gymnasium_env`
Custom Gymnasium environment implementing the fully dynamic BSS rebalancing problem.
- **Environments**: `FullyDynamicEnv-v0`, `StaticEnv-v0`
- **Simulator**: Event-driven bike and truck simulators

### `rl_training`
DDQN agent implementation with training and validation pipelines.
- **CLI**: `bss-train`, `bss-validate`
- **Key modules**: `DQNAgent` (Double DQN), `ReplayBuffer`, `DQN` (GATv2 graph encoder fused with an MLP over the flat environment observation)

### `benchmark`
Baseline comparison tools for evaluating RL performance.
- **CLI**: `bss-benchmark`
- **Strategy**: periodic system-wide TSP-based rebalancing (`StaticEnv`) with a configurable number of events per day

### `results_webapp`
Interactive Dash web application for training visualization.
- **CLI**: `bss-results-webapp`
- **Features**: Real-time monitoring, episode analysis, spatial visualization

---

## 📈 Results Summary

The DDQN agent demonstrated:
- ✅ Adaptation to real-time, location-specific demand fluctuations
- ✅ Reduction in service failures compared to static strategies
- ✅ Feasibility of deep RL for real-time bike rebalancing at scale
- ⚠️ Importance of careful reward design and hyperparameter tuning

While not always outperforming all baselines, the agent proved the viability of RL-based rebalancing and highlighted key challenges in dynamic urban mobility systems.

---

## 🧪 Example: Complete Workflow

```bash
# 1. Preprocess data
bss-preprocess --data-path data/ --source bluebikes

# 2. Train agent
bss-train \
    --data-path data/ \
    --results-path results/ \
    --run-id 0 \
    --num-episodes 100 \
    --device cuda:0

# 3. Monitor training (in another terminal)
bss-results-webapp --results-path results/ --port 8050

# 4. Validate best model
bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type best

# 5. Compare with baselines
bss-benchmark --data-path data/ --results-path results/
```

---

## 🛠️ Development

Each subpackage is independently installable and testable. To modify a package:

1. Navigate to the package directory
2. Make changes
3. Test locally (the `-e` flag ensures changes are immediately reflected)

```bash
cd src/bss_rebalancing/rl_training
# Make changes to src/rl_training/train.py
bss-train --help  # Changes are live!
```

### Tests

`tests/` contains a smoke/regression suite that runs on a tiny synthetic dataset (2x2 cells, generated on the fly, no downloads):
both environments, the DQN agent and replay buffer (the README's custom training loop), `bss-train --one-validation`
end to end, the benchmark CLI, and the preprocessing source registry / trip converter.

```bash
pip install -e . pytest
pytest tests          # ~1-2 minutes on CPU
```

---

## 📚 Citation

If you use this work in your research, please cite:

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

## 📬 Author & Contact


**Edoardo Scarpel**  
Ph.D. Student, University of Padova

For questions, issues, or collaboration:
- **GitHub**: [@edos08](https://github.com/edos08)
- **Issues**: [Create an issue](https://github.com/edos08/BSS-DynamicRebalancing-RL/issues)
- **Email**: edoardo.scarpel@phd.unipd.it

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

This means you are free to:
- ✅ Use the code commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

Under the condition that you include the original copyright and license notice.

---

**Built with ❤️ for sustainable urban mobility**
