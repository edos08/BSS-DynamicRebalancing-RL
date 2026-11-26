# RL Training

Deep Reinforcement Learning training framework for bike-sharing system dynamic rebalancing using Graph Attention Networks (GAT) and Deep Q-Networks (DQN).

This package provides a complete RL training pipeline for learning optimal bike rebalancing policies. It includes:
- **DQN Agent** with Graph Attention Networks for spatial reasoning
- **Experience Replay Buffer** optimized for graph-structured transitions
- **Results Management** with structured logging and model checkpointing
- **CLI Tools** for training and validation

The framework trains agents to control a rebalancing truck navigating a spatial grid, making decisions about bike pickup, drop-off, and charging to minimize system failures.

---

## Installation

Install in editable mode for development:

```bash
cd rl_training
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/rl_training
```

---

## Quick Start

### Training

Basic training with default parameters:

```bash
bss-train --data-path data/ --results-path results/
```

Full training with custom configuration:

```bash
bss-train \
    --data-path data/ \
    --results-path results/ \
    --run-id 1 \
    --num-episodes 150 \
    --max-num-bikes 300 \
    --device cuda:0 \
    --seed 42 \
    --exploration-time 0.6 \
    --log
```

Once epsilon drops below 0.1, every episode snapshot is validated in a background subprocess
(`validate.py`), and the snapshot with the lowest validation `mean_daily_failures` is promoted to `models/best/`.

### Validation

Validate the best model of a run (or a specific one):

```bash
# Best model of run 1
bss-validate --run-id 1 --data-path data/ --results-path results/ --model-type best

# A specific episode snapshot
bss-validate --run-id 1 --data-path data/ --results-path results/ \
    --model-type episode --model-episode 95

# Any checkpoint file
bss-validate --run-id 1 --data-path data/ --results-path results/ \
    --model-path results/run_001/models/best/trained_agent.pt --max-num-bikes 300
```

---

## Package Structure

```
rl_training/
├── README.md
├── pyproject.toml
└── src/
    └── rl_training/
        ├── __init__.py
        ├── train.py                # Training script with CLI
        ├── validate.py             # Validation script with CLI
        ├── utils.py                # Helper functions
        ├── agents/
        │   ├── __init__.py
        │   └── dqn_agent.py        # DQN agent implementation
        ├── memory/
        │   ├── __init__.py
        │   └── replay_buffer.py    # Experience replay buffer
        ├── networks/
        │   ├── __init__.py
        │   └── dqn.py              # GAT-based DQN architecture
        └── results/
            ├── __init__.py
            └── results_manager.py  # Results logging and management
```

---

### Training on the new simulator (`GridZoneEnv`)

```bash
# once: the bundle of the legacy zone (south-east Cambridge, as in sources.json), Cambridge 2024 data
python scripts/clip_osm.py
bss-scenario build configs/scenarios/cambridge-se-2024.yaml --out data/scenarios/cambridge-se-2024
bss-scenario demand data/scenarios/cambridge-se-2024

python -m rl_training.train_grid --bundle data/scenarios/cambridge-se-2024 --results-path results/ \
    --run-id 2 --episodes 30 --device cuda:0
```

Trains the same agent and network on `bss_gym.GridZoneEnv` (issue #166), with the legacy
hyper-parameters. Each transition stores the discounted reward of the action from the env's
per-step rewards (`info["step_rewards"]`), its duration, and `terminal=False` (episodes end by time
limit). The results use the legacy layout (`results/run_NNN/training/episode_NNN/`, the greedy
evaluation on held-out weeks in `validation/`, models in `models/`), so `bss-results-webapp` shows
them next to the legacy runs.

### Checkpoints, resume and best model

Both `bss-train` (legacy env) and `python -m rl_training.train_grid` (GridZoneEnv) save a full training
checkpoint every episode in `results/run_NNN/checkpoints/` (issue #170). It holds the networks, the
optimizer, epsilon, the RNG states and the replay buffer; the last 2 are kept and writes are atomic.

- Add `--resume` to the same command to continue an interrupted run exactly where it stopped. With
  `train_grid`, a resumed run is bit-identical to an uninterrupted one (tested).
- `--checkpoint-every N`, `--keep-checkpoints K`, and `--no-buffer-checkpoint` (smaller files; the
  replay buffer restarts empty) change what is saved.
- `train_grid --stop-after N` stops after N episodes, to train in slices.

**Best model.** `train_grid` validates the greedy policy every `--val-every 5` episodes on a held-out
week (`--val-date`) and keeps the best as `models/model_best.pt`. DQN performance oscillates, so the
last weights are often not the best. `bss-train` does the same with its validation subprocesses
(`models/best/`), unless `--one-validation` is given.

### Weights & Biases

Add `--wandb` (and optionally `--wandb-project <name>`, `--wandb-entity <user or team>`,
`--wandb-mode offline`) to `bss-train` or `train_grid` (issue #171). It logs:
- the configuration, including the reward;
- every episode (return, failures, service level, invalid actions, epsilon, loss, buffer size);
- every validation.

Only the **best model** of each run (by validation) is uploaded, once at the end of training, as one
artifact with alias `best`, and its episode and score as metadata (issue #174). Checkpoints and the
other models stay in `results/run_NNN/`. A resumed run (`--resume`) continues
the same W&B run. Install the extra with `pip install -e "src/bss_rebalancing/rl_training[tracking]"`.

**Settings**: the API key (`WANDB_API_KEY`) and the defaults for the flags (`WANDB_ENTITY`,
`WANDB_PROJECT`, `WANDB_MODE`) come from your personal settings: `bss-settings init`, then fill in
`~/.config/bss-rebalancing/.env` (see [docs/configuration.md](../../../docs/configuration.md)). Flags
override the settings. `wandb login` also works. The code never prints or stores the key. In offline
mode nothing is uploaded; `wandb sync results/run_NNN/wandb/offline-run-*` uploads later.

## Training Parameters

### CLI Arguments

`bss-train`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | int | `0` | Experiment run identifier |
| `--data-path` | str | `"data/"` | Path to preprocessed data directory |
| `--results-path` | str | `"results/"` | Path to save results and models |
| `--device` | str | `"cpu"` | Training device (`cpu`, `cuda:0`, `mps`) |
| `--val-device` | str | same as `--device` | Device for the validation subprocesses |
| `--seed` | int | `42` | Random seed (episode `i` uses `seed + i`) |
| `--num-episodes` | int | `140` | Total training episodes (one week each) |
| `--max-num-bikes` | int | `500` | System bike fleet size |
| `--min-num-bikes` | int | `1` | Minimum bikes per cell at episode start |
| `--enable-repositioning` | flag | — | Distribute extra bikes beyond the per-cell minimum |
| `--use-net-flow` | flag | — | Bias that distribution by predicted net flow (otherwise random) |
| `--exploration-time` | float | `0.6` | Fraction of training after which epsilon is ≈ 0.1 |
| `--critic-mode` | str | `poisson` | `poisson`: the critic score uses a demand forecast sampled from the rates (no look-ahead); `oracle`: it reads the realised future trips |
| `--log` | flag | — | Enable detailed environment logging |
| `--one-validation` | flag | — | Do not validate during training; validate only the final model |

`bss-validate`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | int | `0` | Run whose models/results are used |
| `--data-path` / `--results-path` | str | `data/` / `results/` | Same roots used for training |
| `--model-type` | str | `best` | `best`, `episode` or `final` |
| `--model-episode` | int | — | Episode to load when `--model-type episode` |
| `--model-path` | str | — | Direct path to a `trained_agent.pt` (overrides `--model-type`) |
| `--max-num-bikes` / `--min-num-bikes` | int | `500` / `1` | Fleet size / minimum bikes per cell |
| `--total-timeslots` | int | `56` | Episode length in 3-hour slots |
| `--enable-repositioning` / `--use-net-flow` | flag | — | Initial distribution strategy |
| `--critic-mode` | str | `poisson` | Critic forecast mode; use the one the model was trained with (`bss-train` passes it on) |
| `--num-seed-runs` | int | `1` | Number of validation episodes (seed, seed+1, …) |
| `--device` | str | `cpu` | Device |
| `--non-interactive` | flag | — | Fail instead of prompting if results exist (used by `bss-train`) |

Validation runs with a fixed epsilon of 0.01.

### Default Hyperparameters (`train.py`)

```python
params = {
    "num_episodes": 140,                 # Training episodes
    "batch_size": 64,                    # Replay buffer batch size
    "replay_buffer_capacity": 100000,    # Buffer capacity
    "gamma": 0.95,                       # Discount factor
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "exploration_time": 0.6,             # epsilon_decay is derived from this (see below)
    "lr": 1e-4,                          # Learning rate (SGD)
    "total_timeslots": 56,               # Timeslots per episode (1 week)
    "maximum_number_of_bikes": 500,
    "minimum_number_of_bikes": 1,
    "depot_position_id": 18,             # Depot cell ID
    "validation_epsilon_threshold": 0.1, # Start validating below this epsilon
    "validation_timeout": 600,           # Seconds to wait for a validation subprocess
}
```

- **Epsilon schedule**: `eps(t) = eps_end + (eps_start - eps_end) * exp(-t² / decay)`, with `t` counted in timeslots
  and `decay = T² / ln(10)`, `T = exploration_time * num_episodes * total_timeslots`. At `t = T`, `eps ≈ 0.109`.
- **Target network**: hard update every 8 timeslots (the soft update is disabled in `train_step`; `tau` is currently unused).
- **Depot cell**: `depot_position_id` (18) is hard-coded in `train.py`, `validate.py` and the benchmark; it must exist in your grid
  (check with `bss-preprocess --plot grid-numbered`).

### Reward

The reward is defined in the environment, not in the trainer: see `RewardComponents` in
`gymnasium_env/envs/fully_dynamic_env.py` (base step cost, invalid-action and loop penalties, drop / pick-up / charge terms,
eligibility exploration terms, a term on the change of the global critic score and a final penalty of
`total_failures / total_timeslots`). The constants used in a run are saved to `training/config.json`.

---

## Architecture

### DQN Network

Hybrid Q-network: a Graph Attention encoder over the cell graph, fused with an MLP over the flat environment observation.

#### Graph encoder (one node per cell, edges = adjacent cells with normalized distance)
- Node features (4): `truck_cell`, `critic_score`, `eligibility_score`, `total_bikes`
- 3 × `GATv2Conv` (4→64×4 heads, 256→64×4 heads, 256→128×2 heads, edge dim 1) with ReLU
- `AttentionalAggregation` global pooling → 256
- FC: 256 → 256 → 256 → 128 → 64

#### Agent-state encoder
- Input: the environment observation, of size `17 + 2 × n_cells`
  (5 scalars, 8-dim previous-action one-hot, truck-position one-hot, critical-cell flags, 4 border flags)
- FC: input → 256 → 256 → 128 → 64

#### Fusion
- Concatenate (64 + 64) → FC 256 → 128 → 8 Q-values (one per action)

### DQN Agent

- **Double DQN**: the online network selects the next action, the target network evaluates it
- **Epsilon-greedy** exploration with the schedule above
- **Experience replay** of graph-pair transitions (`PairData`)
- **Optimizer**: SGD, **loss**: Smooth L1, gradient clipping (max norm 10)
- **Semi-MDP target** (#164): an action lasts `tau` steps of 30 s, so
  `y = R + gamma^tau * Q_target(s', argmax_a Q(s', a)) * (1 - terminal)` with `R = sum_i gamma^i r_i` the
  discounted reward collected during the action (`DQNAgent.smdp_target`, `DQNAgent.action_return`).
  `tau` is the duration of the action taken **in `s`**: `ReplayBuffer.push(..., steps=tau, terminal=...)`
  requires it. `terminal` is true only for real terminal states: an episode that ends on a time limit is
  truncated and still bootstraps.
- **Action masking** is supported by `select_action(..., avoid_action=[...])`

### Replay Buffer

`ReplayBuffer` stores `PairData` objects holding the source graph (`x_s`, `edge_index_s`, `edge_attr_s`), the next graph
(`x_t`, …), the agent states, action, reward, done and terminal flags and the duration of the action (`steps`). Batches are built with PyTorch Geometric.
`save_to_files` / `load_from_files` persist it in chunks.

---

## Results Management

The `ResultsManager` provides structured experiment tracking.

### Directory Structure

```
results/
└── run_001/
    ├── training/
    │   ├── config.json                 # Hyperparameters, env defaults, reward constants
    │   ├── train_summary.csv           # One row per episode
    │   ├── logs/train.log
    │   └── episode_000/
    │       ├── scalars.json            # Total reward, failures, mean_daily_failures, ...
    │       ├── timeslot_metrics.csv    # Per-timeslot metrics
    │       ├── step_data.pkl.gz        # Actions, Q-values, reward per action, critic scores
    │       └── cell_subgraph.gpickle   # Per-cell statistics (spatial data)
    ├── validation/
    │   └── episode_095/                # One folder per validated model ("best", "episode_NNN", "custom", ...)
    │       ├── config.json
    │       ├── validation_summary.csv
    │       ├── logs/validate.log
    │       └── episode_000/            # One per --num-seed-runs, same files as training episodes
    └── models/
        ├── episodes/
        │   └── episode_095/
        │       ├── trained_agent.pt    # {'train_model': ..., 'target_model': ...}
        │       └── metadata.json
        ├── best/
        │   ├── trained_agent.pt        # Snapshot with the best validation score
        │   └── metadata.json
        └── best_models_summary.csv
```

### Tracked Metrics

**Episode-level**: total reward, total / mean daily failures, invalid actions, epsilon, mean Q-values

**Timeslot-level**: reward, failures, demand, deployed / depot / truck / traveling / outside bikes, epsilon

**Step-level**: actions, reward per action type, global critic score

**Spatial**: per-cell failures, failure rate, visits, operations, pick-ups, drops, rebalanced count, mean critic, mean bikes

---

## Usage Examples

### Custom Training Loop

The reference implementation is `train_dqn()` in `train.py`. In short:

```python
import gymnasium as gym
import gymnasium_env  # registers the environments
import torch
from torch_geometric.data import Data

from rl_training import (
    DQNAgent, ReplayBuffer, set_seed,
    build_cell_graph_from_cells, update_cell_graph_features, convert_graph_to_data,
)

set_seed(42)
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/", results_path="results/")
agent = DQNAgent(
    num_actions=env.action_space.n,
    observation_space_len=env.observation_space.shape[0],
    replay_buffer=ReplayBuffer(100_000),
    gamma=0.95, lr=1e-4, device="cpu",
)

FEATURES = ["truck_cell", "critic_score", "eligibility_score", "total_bikes"]
obs, info = env.reset(options={"total_timeslots": 56, "maximum_number_of_bikes": 500, "depot_id": 18})
graph = build_cell_graph_from_cells(info["cell_dict"], info["nodes_dict"], info["distance_lookup"])
state = convert_graph_to_data(graph, node_features=FEATURES)
state.agent_state, state.steps = obs, info["steps"]

done = False
while not done:
    single = Data(
        x=state.x, edge_index=state.edge_index, edge_attr=state.edge_attr,
        agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0),
        batch=torch.zeros(state.x.size(0), dtype=torch.long),
    )
    action = agent.select_action(single, epsilon_greedy=True)
    obs, reward, done, timeslot_done, info = env.step(action)

    update_cell_graph_features(graph, info["cell_dict"])
    next_state = convert_graph_to_data(graph, node_features=FEATURES)
    next_state.agent_state, next_state.steps = obs, info["steps"]

    # semi-MDP: this action lasted info["steps"]; time-limit ends are not terminal
    agent.replay_buffer.push(state, action, reward, next_state, done, steps=info["steps"], terminal=False)
    agent.train_step(batch_size=64)
    if timeslot_done:
        agent.update_epsilon()
    state = next_state

env.close()
```

### Load a Trained Model

```python
agent = DQNAgent(num_actions=8, observation_space_len=env.observation_space.shape[0], device="cpu")
agent.load_model("results/run_001/models/best/trained_agent.pt")
agent.train_model.eval()
action = agent.select_action(single_state, greedy=True)   # single_state: a PyG `Data` as above
```

---

## Troubleshooting

### CUDA Out of Memory

**Issue**: Training crashes with CUDA OOM error

**Solutions**:
- Reduce `batch_size` / `replay_buffer_capacity` in `params` (`train.py`)
- Use a separate device for validation: `--val-device cpu`
- Use CPU: `--device cpu`

### Slow Training

**Issue**: Training is very slow

**Causes**:
- Environment logging enabled
- Large graph size (many cells)
- CPU-only training

**Solutions**:
- Disable logging: remove `--log`
- Use GPU: `--device cuda:0`
- Increase cell size during preprocessing

### Model Loading Errors

**Issue**: `FileNotFoundError` or checkpoint mismatch

**Solutions**:
- Verify model path exists
- Check model architecture matches (graph/agent state dimensions)
- Ensure PyTorch version compatibility

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
3. **RL Training** (this package) → Trains agent in simulated environment
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