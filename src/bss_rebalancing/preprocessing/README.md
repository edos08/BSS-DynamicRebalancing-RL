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
bss-preprocess --data-path data/ --source bluebikes
```

This will execute all preprocessing steps:
1. `download`: download and clean the trip data of the chosen source (`sources.json`)
2. `preprocess`: compute Poisson request rates per station pair, day and time slot (and the global rates)
3. `interpolate`: interpolate the sparse rates into PMF matrices
4. `grid`: create the spatial grid (default 300 m cells)
5. `distance`: build the node-to-node distance matrix
6. `nodes`: generate the nearby-nodes dictionary for fast lookups
7. `ev_matrices`: create the truck velocity / consumption tables

When all steps complete, a `utils/preprocessing_manifest.json` is written (source definition, hash of `sources.json`,
configuration, package versions and the command to reproduce the run).

---

## CLI Reference

### Main Command

```bash
bss-preprocess --source SOURCE [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | str | **required** | Source id from `sources.json` (`bluebikes`, `citibike`, ...) |
| `--sources-json` | str | bundled `core/sources.json` | Custom sources file |
| `--data-path` | str | `data/` | Path to data directory |
| `--steps` | str | all | Specific steps to run (comma-separated) |
| `--skip` | str | none | Steps to skip (comma-separated) |
| `--plot` | str | none | Plot mode: `graph`, `grid`, `grid-numbered` (skips preprocessing) |
| `--cell-size` | int | from source | Grid cell size in meters; overrides the source's `cell_size` (300 for the bundled sources) |
| `--verbose` / `-v` | flag | false | Print tracebacks on errors |

### Pipeline Steps

Available steps for `--steps` or `--skip`, in execution order:

1. **`download`** - Download and clean the trip data
2. **`preprocess`** - Compute Poisson rates and global rates ⚠️ *slow*
3. **`interpolate`** - Build PMF matrices
4. **`grid`** - Create spatial grid cells
5. **`distance`** - Build distance matrix
6. **`nodes`** - Create nearby-nodes dictionary
7. **`ev_matrices`** - Generate EV matrices

---

## Data Sources (`sources.json`)

Everything that is specific to a bike-share feed lives in `src/preprocessing/core/sources.json`, one entry per source.
Adding a feed does not require code changes:

```json
"citibike": {
  "base_url": "https://s3.amazonaws.com/tripdata/",
  "url_prefix": "", "url_suffix": "-citibike-tripdata.zip",     // <year><month><suffix>
  "graph_place": "Manhattan, New York, USA",
  "network_type": "bike",
  "bbox": [40.7620, 40.7474, -73.9741, -73.993],                // north, south, east, west
  "years": [2024], "months": [9, 10],
  "station_id_type": "alphanumeric",                            // or "numeric"
  "multi_csv_per_zip": true,
  "column_map": {"started_at": "starttime", "ended_at": "stoptime", ...},  // raw column -> canonical column
  "vital_columns": ["starttime", "stoptime", "start station id", ...],
  "cell_size": 300
}
```

The `download` step downloads the monthly ZIPs, and `core/converter.py` concatenates the raw CSVs, renames columns
through `column_map`, parses the timestamps (many formats), encodes station ids to integers (numeric ids are scaled
by 100, alphanumeric ones are encoded from letters + digits) and writes one clean CSV.

To use a different feed, pass `--sources-json my_sources.json --source my_feed`.

---

## Usage Examples

### Basic Usage

```bash
# Full pipeline for a source
bss-preprocess --data-path data/ --source bluebikes

# Another source
bss-preprocess --data-path data/ --source citibike

# Custom cell size
bss-preprocess --data-path data/ --source bluebikes --cell-size 500

# Custom year / months / bounding box: edit (a copy of) sources.json
bss-preprocess --data-path data/ --source bluebikes --sources-json my_sources.json
```

### Selective Processing

```bash
# Run only specific steps
bss-preprocess --data-path data/ --source bluebikes --steps download,preprocess,grid

# Skip time-consuming preprocessing
bss-preprocess --data-path data/ --source bluebikes --skip preprocess

# Skip download if data already exists
bss-preprocess --data-path data/ --source bluebikes --skip download
```

### Visualization

```bash
# Plot base graph only
bss-preprocess --data-path data/ --source bluebikes --plot graph

# Plot with grid overlay
bss-preprocess --data-path data/ --source bluebikes --plot grid

# Plot with grid and cell IDs
bss-preprocess --data-path data/ --source bluebikes --plot grid-numbered
```

Plots are saved to `data/plots/`.

### Advanced Configuration

```bash
# Verbose output for debugging
bss-preprocess --data-path data/ --source bluebikes --verbose
```

Process a different period, another area or a larger cell size by editing the corresponding entry of `sources.json`
(`years`, `months`, `bbox`, `graph_place`, `cell_size`).

---

## Output Structure

After running the pipeline, your data directory will contain:

```
data/
├── trips/                                  # Downloaded ZIPs + one cleaned CSV
│   ├── 202209-bluebikes-tripdata.zip
│   ├── 202210-bluebikes-tripdata.zip
│   └── bluebikes-09-10-tripdata.csv
│
├── rates/                                  # Poisson rates
│   └── 09-10/                              # Month range
│       ├── monday/
│       │   ├── 00-extracted-rates.csv      # Timeslot 0 (01:00-04:00)
│       │   └── ...                         # Timeslots 01-07
│       └── ...                             # tuesday ... sunday
│
├── matrices/                               # PMF matrices (used by the environments)
│   └── 09-10/
│       ├── monday/
│       │   ├── 00-pmf-matrix.csv
│       │   └── ...
│       └── ...
│
├── utils/                                  # Processed utilities
│   ├── network.graphml                     # Street network graph
│   ├── cell_data.pkl                       # Spatial grid cells
│   ├── distance_matrix.csv                 # Node-to-node distances
│   ├── ev_consumption_matrix.csv
│   ├── ev_velocity_matrix.csv
│   ├── global_rates.pkl                    # Global rate per (day, timeslot)
│   ├── nearby_nodes.pkl                    # Node neighborhood data
│   └── preprocessing_manifest.json         # Reproducibility manifest
│
└── plots/                                  # Visualizations (--plot)
    ├── graph.png
    ├── grid.png
    └── grid_numbered.png
```

Note: the environments currently look for `matrices/09-10` (see `gymnasium_env/simulator/utils.py`), so months other than
`[9, 10]` require updating that path.

---

## Upstream Data

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

Run-level settings live in `PreprocessingConfig` (`config.py`); source-specific ones (place, bbox, period, columns,
cell size) live in `sources.json`.

```python
@dataclass
class PreprocessingConfig:
    source_id = "bluebikes"
    sources_json = <bundled core/sources.json>
    network_type = "bike"
    data_path = "data/"
    nodes_to_remove = []              # graph nodes to drop
    days_of_week = ["Monday", ..., "Sunday"]
    num_time_slots = 8                # 3-hour slots
    cell_size = 300                   # meters (from the source unless --cell-size is given)
    interpolation_radius = 500        # meters, PMF interpolation
    user_radius = 250                 # meters, nearby nodes
```

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
        │   ├── sources.json    # Data-source registry
        │   ├── sources.py      # Source dataclass + loader
        │   ├── converter.py    # Raw trip CSV -> clean CSV (columns, timestamps, station ids)
        │   ├── manifest.py     # Reproducibility manifest
        │   ├── graph.py        # Network graph operations
        │   ├── grid.py         # Spatial grid operations
        │   ├── plotting.py     # Visualization functions
        │   └── utils.py        # Helper functions
        │
        └── steps/           # Pipeline steps
            ├── __init__.py
            ├── download_trips.py              # download
            ├── preprocess_data.py             # preprocess (rates + global rates)
            ├── interpolate_data.py            # interpolate (PMF matrices)
            ├── preprocess_truck_grid.py       # grid
            ├── preprocess_distance_matrix.py  # distance
            ├── preprocess_nodes_dictionary.py # nodes
            └── create_ev_matrices.py          # ev_matrices
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
# Process fewer months at a time: set "months" in sources.json

# Or use larger cells to create fewer of them
bss-preprocess --data-path data/ --source bluebikes --cell-size 500
```

### Download Failures

If BlueBikes data download fails:

1. Verify that the month/year exists under the source's `base_url` (`sources.json`)
2. Try downloading manually and place the ZIPs in `data/trips/`, then re-run with `--steps download` (existing ZIPs are reused)

### Graph Building Errors

If graph initialization fails:

```bash
# Try a larger "bbox" for the source in sources.json, then re-run the pipeline
bss-preprocess --data-path data/ --source bluebikes
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
