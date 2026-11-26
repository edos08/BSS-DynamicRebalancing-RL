# Results WebApp

Interactive web dashboard for real-time monitoring and visualization of BSS RL training and validation results.

This package provides a **Dash-based web application** for visualizing and analyzing results from the RL training pipeline. The dashboard offers real-time monitoring, interactive plots, and comprehensive analysis of training metrics, enabling researchers to track model performance and debug training issues.

### Key Features

- **Real-time Monitoring**: Auto-refresh every 5 seconds during training
- **Dual Mode Visualization**: Switch between training and validation results
- **Multi-Run Comparison**: Compare different experimental runs
- **Episode-Level Analysis**: Drill down into specific episodes
- **Interactive Plots**: Powered by Plotly for zoom, pan, and hover details
- **Export Capability**: Download plots as high-resolution SVG images

---

## Installation

Install in editable mode for development:

```bash
cd results_webapp
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/results_webapp
```

---

## Quick Start

### Launch Dashboard

```bash
bss-results-webapp --results-path results/ --port 8050
```

Then open your browser to: **http://localhost:8050**

### Custom Configuration

```bash
bss-results-webapp \
    --results-path /path/to/results \
    --port 8080 \
    --update-interval 10000 \
    --debug
```

---

## Package Structure

```
results_webapp/
├── README.md
├── pyproject.toml
└── src/
    └── results_webapp/
        ├── __init__.py
        ├── app.py              # Main Dash application
        ├── callbacks.py        # Interactive callback functions
        ├── data_loader.py      # Data loading utilities
        ├── layouts.py          # UI layout components
        └── plotting.py         # Plotly plotting functions
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results-path` | str | `"../results"` | Path to results directory |
| `--data-path` | str | none | Data directory (needs `utils/network.graphml`); enables the spatial heatmaps |
| `--port` | int | `8050` | Port to run server on |
| `--update-interval` | int | `300000` | Auto-refresh interval (milliseconds, i.e. 5 minutes) |
| `--debug` | flag | — | Enable debug mode (auto-reload) |

---

## Dashboard Layout

### Navigation Controls

**Top Control Panel**:
- **Run Selector**: Choose from available experimental runs
- **Mode Selector**: Switch between Training and Validation
- **Episode Selector**: Select specific episode for detailed analysis

### Overview Tab (📊)

**Training and Validation Metrics**:

1. **Failures per Timeslot**: Total system failures with cumulative mean
2. **Rewards per Timeslot**: Agent rewards with cumulative trend
3. **Epsilon Decay**: Exploration rate evolution over episodes
4. **Deployed Bikes**: Number of bikes in the system per timeslot

**Training-Only Metrics**:

5. **Q-Values**: Mean Q-values across all actions per timeslot
6. **Global Critic Score**: System-level health metric over steps
7. **Training Loss**: TD error loss with moving average

### Episode Details Tab (🔍)

**Episode Statistics Cards**:
- Total Failures
- Total Reward
- Mean Failures
- Epsilon Value

**Detailed Episode Plots**:

1. **Rewards per Timeslot**: Episode-specific reward progression
2. **Failures per Timeslot**: Episode-specific failure counts
3. **Action Distribution**: Bar chart of action frequency
4. **Reward per Action**: Mean reward for each action type

---

## Usage Examples

### Monitor Live Training

While training is running:

```bash
# Terminal 1: Start training
cd rl_training
bss-train --data-path data/ --results-path ../results --run-id 1

# Terminal 2: Launch dashboard
cd results_webapp
bss-results-webapp --results-path ../results --update-interval 5000
```

The dashboard will auto-refresh every 5 seconds, showing the latest results.

### Analyze Completed Runs

```bash
# Launch dashboard for multiple completed runs
bss-results-webapp --results-path results/ --port 8050
```

Use the **Run Selector** to switch between different experiments.

### Export Plots

1. Hover over any plot
2. Click the **📷 camera icon** in the toolbar
3. Select SVG format for publication-quality images
4. Plots are saved as `plot.svg` (configurable in `layouts.py`)

### Debug Mode

For development with auto-reload:

```bash
bss-results-webapp --results-path results/ --debug
```

---

## Data Loading

### Supported Data Formats

The webapp automatically loads data from the `rl_training` results structure:

```
results/
└── run_001/
    ├── config.json                  # Hyperparameters
    ├── training/
    │   ├── training_summary.csv     # Episode-level aggregates
    │   └── episode_000/
    │       ├── scalars.json         # Total reward, failures, etc.
    │       ├── timeslot_metrics.csv # Per-timeslot data
    │       ├── step_data.pkl.gz     # Actions, Q-values, losses
    │       └── cell_subgraph.gpickle
    └── validation/
        └── ...
```

### Data Loading Functions

**Run Discovery**:
```python
from results_webapp import discover_runs, load_run_config

runs = discover_runs(Path("results/"))
config = load_run_config(Path("results/run_001"))
```

**Episode Data**:
```python
from results_webapp import load_episode_data, get_available_episodes

episodes = get_available_episodes(Path("results/run_001"), "training")
data = load_episode_data(Path("results/run_001"), "training", episode=0)
```

**Concatenated Time Series**:
```python
from results_webapp import load_concatenated_timeslot_data

# Load failures across all episodes
failures = load_concatenated_timeslot_data(
    Path("results/run_001"),
    mode="training",
    metric="failures"
)
```

---

## Troubleshooting

### Dashboard Won't Start

**Issue**: `Address already in use`

**Solution**: Change port or kill existing process:
```bash
# Use different port
bss-results-webapp --port 8080

# Or kill process on port 8050
lsof -ti:8050 | xargs kill -9
```

### No Runs Displayed

**Issue**: Run selector is empty

**Causes**:
- Incorrect `--results-path`
- No `run_*` directories found

**Solution**:
```bash
# Check results directory structure
ls -la results/

# Ensure correct path
bss-results-webapp --results-path /absolute/path/to/results
```

### Plots Not Updating

**Issue**: Data doesn't refresh during training

**Solution**:
- Check auto-refresh interval (CLI default: 300000 ms = 5 minutes; lower it with `--update-interval 5000`)
- Verify training is writing files to correct location
- Look for errors in browser console (F12)

### Missing Validation Data

**Issue**: "No validation data available"

**Cause**: No validation has been run yet

**Solution**:
- Run validation: `bss-validate --run-id N --data-path data/ --results-path results/ --model-type best`
- Or wait for periodic validation during training

### Slow Performance

**Issue**: Dashboard is laggy with many episodes

**Solutions**:
- Increase the update interval: `--update-interval 600000` (10 minutes)
- Use summary CSVs instead of loading individual episodes
- Filter episodes in callbacks (reduce data volume)

---

## Advanced Features

### Custom Port Configuration

Run multiple dashboards simultaneously. `--results-path` must point to a directory that *contains* `run_NNN` folders
(not to a single run folder), so use separate result roots:

```bash
# Dashboard for experiment A
bss-results-webapp --results-path results_a/ --port 8050

# Dashboard for experiment B (different terminal)
bss-results-webapp --results-path results_b/ --port 8051
```

### Programmatic Usage

Create custom dashboards:

```python
from results_webapp.app import create_app

app = create_app(
    results_path="results/",
    data_path="data/",          # optional, enables the spatial heatmaps
    port=8050,
    update_interval_ms=5000     # the CLI default is 300000
)

app.run(debug=True, host='0.0.0.0', port=8050)
```

### Custom Callbacks

Add new plots by extending `callbacks.py`:

```python
from dash import Input, Output
import plotly.graph_objects as go

@app.callback(
    Output('my-custom-plot', 'figure'),
    Input('run-selector', 'value')
)
def update_custom_plot(run_path):
    # Load data
    # Create figure
    return fig
```

---

## Integration with Training Pipeline

### Typical Workflow

1. **Start Training**:
   ```bash
   bss-train --data-path data/ --results-path results/ --run-id 1
   ```

2. **Launch Dashboard** (different terminal):
   ```bash
   bss-results-webapp --results-path results/
   ```

3. **Monitor Training**:
   - Watch failures decrease
   - Track epsilon decay
   - Observe Q-value stabilization
   - Check loss convergence

4. **Analyze Results**:
   - Switch to validation mode after training completes
   - Compare training vs validation performance
   - Export plots for reports

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
3. **RL Training** → Trains agent in simulated environment
4. **Benchmark** → Compares against baselines
5. **Results WebApp** (this package) → Visualizes training progress

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
