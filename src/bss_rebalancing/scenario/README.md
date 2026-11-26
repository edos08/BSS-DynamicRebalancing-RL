# bss-scenario

Builds **scenario bundles** for the BSS simulator v2: cleaned trips, networks, stations, demand
model and truck travel data, written as Parquet / NumPy files with a documented schema
(`docs/sim-v2/SPEC.md` §4). The simulator only reads bundles; this package never imports the
simulator or the legacy thesis packages.

## Install

```bash
pip install -e src/bss_rebalancing/scenario
bss-scenario --help
```

## Commands

### `bss-scenario build`: the whole bundle

```bash
bss-scenario build configs/scenarios/cambridge-2024.yaml --out data/scenarios/cambridge-2024
bss-scenario check data/scenarios/cambridge-2024
```

Runs `trips`, `networks` and `stations` from one YAML config (paths relative to the YAML), then
writes `scenario.yaml` and `manifest.json` (inputs and outputs with SHA-256, versions, command
line). `check` verifies the schema version, that every component is complete, that only
Parquet/NumPy/JSON/YAML files are present (no pickles) and, unless `--no-hashes`, every file hash.
In Python: `bss_scenario.bundle.load_bundle(path, verify_hashes=False)`.
Cambridge 2024: 25 files, 182 MB, 15 s.

The steps can also be run one by one:

### `bss-scenario trips`: clean recorded trips

```bash
bss-scenario trips \
  --raw data/raw/bluebikes/2024 \
  --boundary data/raw/osm/cambridge_boundary.geojson \
  --out data/scenarios/cambridge-2024/trips
```

Reads the raw monthly BlueBikes CSVs (2024 schema), localises timestamps to America/New_York
(DST-safe), drops month-boundary duplicates, flags problems, marks which ends lie inside the city
and writes `YYYY-MM.parquet` per local start month plus `cleaning_report.json` (inputs with
SHA-256, configuration, counts per flag and month, outputs with SHA-256). Options:
`--min-duration` (60 s), `--max-duration` (10800 s), `--keep touching_city|all`.

Rows are **flagged, never deleted** (except trips not touching the city with the default `--keep`).
`valid` = no flag set.

| Column | Type | Meaning |
|--------|------|---------|
| `trip_id` | str | operator `ride_id` |
| `month` | str | `YYYY-MM`, local start month |
| `bike_type` | enum `classic`/`electric` | |
| `user_type` | enum `member`/`casual` | |
| `start_time`, `end_time` | datetime[ms, America/New_York] | |
| `start_station_id`, `start_station_name`, `end_station_id`, `end_station_name` | str, nullable | null when the bike was taken/left outside a station |
| `start_lat`, `start_lon`, `end_lat`, `end_lon` | f64, end nullable | WGS84 |
| `duration_s` | f64 | |
| `start_in_city`, `end_in_city` | bool (end nullable) | inside the boundary polygon; one end outside = `EXTERNAL` trip |
| `flag_negative` | bool | ends before it starts (clock glitch) |
| `flag_short` | bool | `duration_s` < min duration |
| `flag_long` | bool | `duration_s` > max duration |
| `flag_no_end` | bool | no end coordinates (bike not returned) |
| `valid` | bool | no flag set |

Python: `bss_scenario.trips.read_trips(paths, valid_only=False)` reads and checks the tables
(`TRIPS_SCHEMA`, `TRIPS_SCHEMA_VERSION`).

### `bss-scenario demand`: demand model

```bash
bss-scenario demand data/scenarios/cambridge-2024     # (re)fit demand/ in an existing bundle
```

Part of `build`. Maps trip ends to places (in-city stations + `EXTERNAL`), fits the daily volume
(month × day group × log-normal noise), and selects the day grouping and smoothing strengths of
the hourly OD shape on held-out days, then writes `demand/` (places, volume, `shape.npz`, report
with the full selection table). Cambridge 2024: `day_types`, alpha 1024, beta 512.

### `bss-scenario networks` and `bss-scenario stations`

```bash
bss-scenario networks --bike data/raw/osm/cambridge_bike.graphml --drive data/raw/osm/cambridge_drive.graphml \
  --out data/scenarios/cambridge-2024/network
bss-scenario stations --trips data/scenarios/cambridge-2024/trips --boundary data/raw/osm/cambridge_boundary.geojson \
  --gbfs data/raw/gbfs/2026-11-25/station_information.json --network data/scenarios/cambridge-2024/network \
  --out data/scenarios/cambridge-2024/stations/stations.parquet
```

`networks`: node/edge tables (largest strongly connected component, shared metre projection),
`bike_dist.npy`, walking neighbours, free-flow `drive_time.npy` (no busways), bike→drive mapping.
`stations`: stations seen in valid trips with modal position, usage, GBFS capacity (dated
snapshot, median fallback), `in_city` and nearest bike node.

### `bss-scenario traffic`: truck travel-time data

```bash
bss-scenario traffic sample  --graph data/raw/osm/cambridge_drive.graphml --out data/raw/traffic/od_sample.parquet
export TOMTOM_API_KEY=...   # never commit it
bss-scenario traffic collect --od data/raw/traffic/od_sample.parquet --cache data/raw/traffic/tomtom --daily
bss-scenario traffic status  --od data/raw/traffic/od_sample.parquet --cache data/raw/traffic/tomtom
```

`sample` draws OD pairs stratified by distance band and area. `collect` asks the TomTom Routing
API for the no-traffic and historic-traffic travel time of every pair at every hour of a
reference weekday, Saturday and Sunday (`plan.json` in the cache fixes the dates). Responses are
cached as JSON (never the key); reruns only query what is missing; `--daily` runs one batch of
`--max-requests` (2,400, under the free 2,500/day) per day until done.

## Status

Milestone M1 (epic #38). Commands are added as they are implemented.
