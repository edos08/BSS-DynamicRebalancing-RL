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

## Status

Milestone M1 (epic #38). Commands are added as they are implemented.
