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

📄 Publication
Paper: E. Scarpel, Fully Dynamic Rebalancing of Dockless Bike-Sharing Systems using Deep Reinforcement Learning, arXiv:2605.14501, 2025. https://arxiv.org/abs/2605.14501

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
        │       ├── core/              # Graph, grid, plotting utilities
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
# Full pipeline
bss-preprocess --data-path data/

# Specific steps only
bss-preprocess --data-path data/ --steps download,preprocess,grid

# Skip certain steps
bss-preprocess --data-path data/ --skip download

# Visualize the grid
bss-preprocess --data-path data/ --plot grid-numbered
```

**Key arguments**:
- `--data-path`: Data directory (default: `data/`)
- `--year`: Year to process (default: 2022)
- `--months`: Comma-separated months (default: `9,10`)
- `--cell-size`: Grid cell size in meters (default: 300)
- `--steps`: Run specific steps only
- `--plot`: Visualization mode (`graph`, `grid`, `grid-numbered`)

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
    --num-bikes 400 \
    --exploration-time 0.7 \
    --device cuda:0 \
    --seed 42
```

**Key arguments**:
- `--run-id`: Experiment identifier (default: 0)
- `--num-episodes`: Training episodes (default: 140)
- `--num-bikes`: Fleet size (default: 500)
- `--exploration-time`: Fraction of episodes for exploration (default: 0.6)
- `--device`: Hardware device (`cpu`, `cuda:0`, `mps`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--one-validation`: Validate only at the end

### 3. Validate the Model

Test a trained model on validation data:

```bash
bss-validate \
    --model-path results/run_000/models/best_model.pt \
    --data-path data/ \
    --epsilon 0.05 \
    --total-timeslots 56
```

**Key arguments**:
- `--model-path`: Path to trained model (required)
- `--epsilon`: Exploration rate for validation (default: 0.05)
- `--total-timeslots`: Episode length (default: 56 = 1 week)
- `--run-id`: Validation run identifier (default: 999)

### 4. Run Benchmarks

Compare against baseline strategies:

```bash
# Basic benchmark
bss-benchmark --data-path data/

# With custom parameters
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --num-episodes 5 \
    --maximum-number-of-bikes 400
```

**Key arguments**:
- `--data-path`: Path to data folder (required)
- `--results-path`: Where to save results (default: current directory)
- `--num-episodes`: Number of episodes to simulate (default: 1)
- `--maximum-number-of-bikes`: Fleet size (default: 300)

### 5. Visualize Training Progress

Launch the interactive dashboard to monitor training in real-time:

```bash
bss-results-webapp --results-path results/ --port 8050
```

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
- **CLI**: `bss-preprocess`
- **Key modules**: `download_trips`, `preprocess_data`, `preprocess_truck_grid`

### `gymnasium_env`
Custom Gymnasium environment implementing the fully dynamic BSS rebalancing problem.
- **Environments**: `FullyDynamicEnv-v0`, `StaticEnv-v0`
- **Simulator**: Event-driven bike and truck simulators

### `rl_training`
DDQN agent implementation with training and validation pipelines.
- **CLI**: `bss-train`, `bss-validate`
- **Key modules**: `DQNAgent`, `ReplayBuffer`, `DuelingDQN`

### `benchmark`
Baseline comparison tools for evaluating RL performance.
- **CLI**: `bss-benchmark`
- **Strategies**: Static, naive, demand-based rebalancing

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
bss-preprocess --data-path data/ --year 2022 --months 9,10

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
bss-validate \
    --model-path results/run_000/models/best_model.pt \
    --data-path data/

# 5. Compare with baselines
bss-benchmark --data-path data/
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
