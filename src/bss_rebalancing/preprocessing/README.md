# BSS Preprocessing Pipeline

Data preprocessing pipeline for the BSS Dynamic Rebalancing RL project.

This package handles the complete data pipeline from raw BlueBikes trip data to processed rate matrices, spatial grids, and network graphs ready for RL training.

---

## Installation

Install in editable mode for development:

```bash
cd preprocessing
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/preprocessing
```

---

## Quick Start

### Run Full Pipeline

Download and preprocess all data in one command:

```bash
bss-preprocess --data-path data/
```

This will execute all preprocessing steps:
1. Download BlueBikes trip data
2. Compute Poisson request rates per station pair
3. Interpolate sparse data using PMF matrices
4. Create spatial grid (300m cells)
5. Build distance matrices with TomTom traffic
6. Compute global rate statistics
7. Generate nodes dictionary for fast lookups
8. Create electric vehicle (EV) consumption matrices

---

## CLI Reference

### Main Command

```bash
bss-preprocess [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data-path` | str | `data/` | Path to data directory |
| `--year` | int | `2022` | Year of data to process |
| `--months` | str | `9,10` | Comma-separated months |
| `--cell-size` | int | `300` | Grid cell size in meters |
| `--steps` | str | all | Specific steps to run (comma-separated) |
| `--skip` | str | none | Steps to skip (comma-separated) |
| `--plot` | str | none | Plot mode: `graph`, `grid`, `grid-numbered` |
| `--bbox` | str | `[42.370, 42.353, -71.070, -71.117]` | Bounding box [N,S,E,W] |
| `--verbose` / `-v` | flag | false | Enable verbose output |

### Pipeline Steps

Available steps for `--steps` or `--skip`:

1. **`download`** - Download BlueBikes trip data
2. **`preprocess`** - Compute Poisson rates ⚠️ *slow*
3. **`interpolate`** - Build PMF matrices
4. **`grid`** - Create spatial grid cells
5. **`distance`** - Build distance matrix
6. **`rates`** - Compute global statistics
7. **`nodes`** - Create nodes dictionary
8. **`ev_matrices`** - Generate EV matrices

---

## Usage Examples

### Basic Usage

```bash
# Full pipeline with defaults
bss-preprocess --data-path data/

# Process specific year and months
bss-preprocess --data-path data/ --year 2022 --months 9,10,11

# Custom cell size
bss-preprocess --data-path data/ --cell-size 500
```

### Selective Processing

```bash
# Run only specific steps
bss-preprocess --data-path data/ --steps download,preprocess,grid

# Skip time-consuming preprocessing
bss-preprocess --data-path data/ --skip preprocess

# Skip download if data already exists
bss-preprocess --data-path data/ --skip download
```

### Visualization

```bash
# Plot base graph only
bss-preprocess --data-path data/ --plot graph

# Plot with grid overlay
bss-preprocess --data-path data/ --plot grid

# Plot with grid and cell IDs
bss-preprocess --data-path data/ --plot grid-numbered
```

Plots are saved to `data/plots/`.

### Advanced Configuration

```bash
# Custom bounding box (Cambridge area)
bss-preprocess \
    --data-path data/ \
    --bbox "[42.400, 42.350, -71.050, -71.150]" \
    --cell-size 400

# Verbose output for debugging
bss-preprocess --data-path data/ --verbose

# Process entire year
bss-preprocess \
    --data-path data/ \
    --year 2022 \
    --months 1,2,3,4,5,6,7,8,9,10,11,12
```

---

## Output Structure

After running the pipeline, your data directory will contain:

```
data/
├── trips/                                  # Downloaded trip data
│   ├── 202209-bluebikes-tripdata.csv
│   └── 202210-bluebikes-tripdata.csv
│
├── rates/                                  # Poisson rate matrices
│   └── 09-10/                              # Month range
│       ├── 00/                             # Timeslot 0 (01:00-04:00)
│       │   ├── monday-poisson-rates.csv
│       │   ├── tuesday-poisson-rates.csv
│       │   └── ...
│       ├── 01/                             # Timeslot 1 (04:00-07:00)
│       └── ...                             # Timeslots 2-7
│
├── matrices/                               # Rate matrices
│   └── 09-10/
│       ├── 00/
│       │   ├── monday-rate-matrix.csv
│       │   └── ...
│       └── ...
│
├── utils/                                  # Processed utilities
│   ├── network.graphml           # Street network graph
│   ├── cell_data.pkl                       # Spatial grid cells
│   ├── distance_matrix.csv                 # Travel time matrix
│   ├── ev_consumption_matrix.csv           # Travel time matrix
│   ├── ev_velocity_matrix.csv              # Travel time matrix
│   ├── global_rates.pkl                    # Global statistics
│   ├── nearby_nodes.pkl                    # Node neighborhood data
│
└── plots/                                  # Visualizations
    ├── graph.png
    ├── grid.png
    └── grid_numbered.png
```

---

## Data Sources

### BlueBikes Trip Data
- **Source**: [BlueBikes System Data](https://www.bluebikes.com/system-data)
- **Format**: CSV files with trip records
- **Fields**: start/end stations, timestamps, coordinates
- **Coverage**: Cambridge, MA metropolitan area

### OpenStreetMap Network
- **Source**: [OSM via OSMnx](https://osmnx.readthedocs.io/)
- **Type**: Bike-friendly street network
- **Area**: Cambridge, MA (configurable bounding box)

### TomTom Traffic Data
- **Purpose**: Time-dependent travel speeds
- **Integration**: Via distance matrix preprocessing
- **Effect**: Realistic travel time estimates

---

## Configuration

Default configuration in `config.py`:

```python
@dataclass
class PreprocessingConfig:
    # Location
    place = ["Cambridge, Massachusetts, USA"]
    network_type = "bike"
    bbox = (42.370, 42.353, -71.070, -71.117)  # N, S, E, W

    # Time
    year = 2022
    months = [9, 10]  # September, October
    days_of_week = ["Monday", ..., "Sunday"]
    num_time_slots = 8  # 3-hour slots

    # Grid
    cell_size = 300  # meters
    interpolation_radius = 500  # meters
    user_radius = 250  # meters

    # Paths
    data_path = "data/"
```

Customize via CLI arguments or by modifying `config.py`.

---

## Package Structure

```
preprocessing/
├── pyproject.toml           # Package configuration
├── README.md                # This file
└── src/
    └── preprocessing/
        ├── __init__.py
        ├── __main__.py      # Entry: python -m preprocessing
        ├── cli.py           # CLI interface
        ├── config.py        # Configuration dataclass
        │
        ├── core/            # Core utilities
        │   ├── __init__.py
        │   ├── graph.py     # Network graph operations
        │   ├── grid.py      # Spatial grid operations
        │   ├── plotting.py  # Visualization functions
        │   └── utils.py     # Helper functions
        │
        └── steps/           # Pipeline steps
            ├── __init__.py
            ├── download_trips.py              # Step 1
            ├── preprocess_data.py             # Step 2
            ├── interpolate_data.py            # Step 3
            ├── preprocess_truck_grid.py       # Step 4
            ├── preprocess_distance_matrix.py  # Step 5
            ├── preprocess_global_rates.py     # Step 6
            ├── preprocess_nodes_dictionary.py # Step 7
            └── create_ev_matrices.py          # Step 8
```

---

## Troubleshooting

### Missing Dependencies

If you encounter import errors:

```bash
pip install osmnx networkx geopandas matplotlib
```

### Memory Issues

The `preprocess` step can be memory-intensive. If it fails:

```bash
# Process fewer months at a time
bss-preprocess --data-path data/ --months 9

# Or reduce cell size to create fewer cells
bss-preprocess --data-path data/ --cell-size 500
```

### Download Failures

If BlueBikes data download fails:

1. Verify month/year exists in `https://s3.amazonaws.com/hubway-data/`
2. Try downloading manually and place in `data/trips/`

### Graph Building Errors

If graph initialization fails:

```bash
# Try with a larger bounding box
bss-preprocess --data-path data/ --bbox "[42.400, 42.300, -71.000, -71.200]"
```

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** (this package) → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
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
