# BSS Simulator v2 — Specification (draft for review)

Status: **draft** · Branch: `sim-v2` · Date: 2026-11-25

This document specifies a new, independent simulator for bike-sharing systems (BSS) and the
tooling around it. It replaces the simulation part of `gymnasium_env`; the current packages stay
frozen so that the thesis results remain reproducible.

---

## 1. Goals and non-goals

**Goals**

1. **A simulator that knows nothing about RL.** It simulates a city, its users, its fleet and its
   operators. Reinforcement learning, heuristics and baselines are *clients* of the same API.
2. **Operators are commanded units.** Trucks do nothing until told; each has a command queue; the
   simulator reports when a truck becomes idle (StarCraft-like).
3. **Realism where it changes the results**: correct demand rates, realistic users (walking,
   abandonment), trucks on the road network with service times, batteries, docked *and* dockless.
4. **No information leaks by construction.** Controllers see the state and an explicit forecast,
   never the simulator's future events.
5. **Fast.** Budget in §12: a simulated week of Cambridge in about a second, so that training
   time is dominated by the learner, not the simulator.
6. **Reproducible.** `reset(seed)` gives the same run regardless of what ran before.
7. **Pluggable models**: demand estimation, spatial spreading, users, energy, travel times.

**Non-goals (for now)**

- Long-term demand dynamics (induced demand, pricing).
- Microscopic traffic simulation.
- A GUI (the event log feeds the existing results web app or a future viewer).

---

## 2. Agreed decisions

| # | Decision |
|---|----------|
| D1 | Trucks are externally commanded, idle until commanded; per-truck command queue; `TruckIdle` notification. |
| D2 | The core uses domain commands (go to, pick up, drop, swap battery…). The old zone-by-zone grid moves are an **adapter**. |
| D3 | Several trucks supported, set at construction; default 1. |
| D4 | Spatial zones (square grid) are an adapter-level **view**, not part of the simulator. |
| D5 | Python first (NumPy); then a Rust engine behind the same API, compared on the same benchmark. |
| D6 | Strict reproducibility. |
| D7 | Both **docked** and **dockless** operation. |
| D8 | Demand estimation and spatial spreading (the old "interpolation") are separate pluggable components. |
| D9 | Fleet composition configurable: all-electric, all-mechanical or a share of e-bikes. |
| D10 | Demand time resolution: **1 hour**. |
| D11 | Two demand generators: **probabilistic** (fitted model) and **replay** (recorded requests). |
| D12 | Data: BlueBikes **2024**, full city of **Cambridge, MA** (OSM bike and drive networks). |
| D13 | Truck travel times depend on historical traffic (TomTom when available) and trucks consume energy; both are pluggable models. |
| D14 | Development is tracked on GitHub: issues, milestones, PRs into `dev`, CI (see `CONTRIBUTING.md`). |

---

## 3. Architecture

```
raw data ──► bss_scenario ──► scenario bundle (files) ──► bss_sim ◄── controllers
(trips, OSM)   builder           versioned, no pickles      engine      ├─ bss_gym  (RL adapters: grid zones, obs, reward)
                                                                       ├─ bss_policies (static TSP, heuristics, do-nothing)
                                                                       └─ scripts / notebooks
```

| Package | Responsibility | Depends on |
|---------|----------------|------------|
| `bss_scenario` | Clean trips, build networks, estimate demand, spread it spatially, write a **scenario bundle**. CLI `bss-scenario`. | polars, osmnx, numpy, scipy |
| `bss_sim` | Discrete-event engine, state, models, operators, information layer, telemetry. **No RL, no Gym.** | numpy (+ pyarrow for logs) |
| `bss_gym` | Gymnasium environments built on `bss_sim`: zone views, observations, rewards, action adapters. | bss_sim, gymnasium |
| `bss_policies` | Baselines as controllers: static TSP, greedy, do-nothing. | bss_sim |

Rules:
- `bss_sim` never imports `bss_gym` or `bss_policies`; `bss_scenario` never imports `bss_sim`
  (the old `preprocessing → gymnasium_env.Cell` pickle coupling disappears).
- Data crosses package boundaries only as files with a documented schema (§4) or as plain
  dataclasses/arrays — never as pickled classes.

---

## 4. Scenario bundle

A scenario is a directory produced by `bss-scenario build`; the simulator only reads it.

```
scenarios/cambridge-2024/
├── scenario.yaml              # id, schema_version, source, fit period, timezone, slot length, hashes
├── manifest.json              # how it was built (inputs, versions, command line)
├── network/
│   ├── bike_nodes.parquet     # node_idx, osm_id, lat, lon, x, y (projected metres)
│   ├── bike_dist.npy          # float32 [N, N] shortest-path metres on the (directed) bike network
│   ├── bike_walk_nbrs.npz     # CSR: for each node, nodes within the walking radius, sorted by distance
│   ├── drive_nodes.parquet
│   ├── drive_time.npy         # float32 [M, M] free-flow driving time (s), one-way streets respected
│   ├── drive_congestion.parquet  # factor per (weekday, hour): time = free_flow × factor
│   └── bike_to_drive.npy      # int32 [N]: nearest drive node of each bike node (+ distance)
├── stations/stations.parquet  # station_id, name, lat, lon, capacity, bike_node
├── demand/
│   ├── shape.npz              # sparse OD rates per slot (station level, incl. EXTERNAL)
│   ├── volume.parquet         # per date: observed volume factor; monthly levels; day-noise params
│   ├── spread.npz             # CSR: station -> distribution over bike nodes (dockless spreading)
│   └── durations.parquet      # empirical trip speed / duration model parameters
└── trips/
    └── 2024-MM.parquet        # cleaned recorded trips for replay (one file per month)
```

Sizes for Cambridge (5,017 bike nodes, 1,993 drive nodes, 219 stations):
`bike_dist` 100 MB, `drive_time` 16 MB, demand shape ≈ 219² × 168 slots ≈ 8 M values (much less sparse-stored).
No N² demand matrices are ever materialised.

`schema_version` is checked at load; a mismatch is an error with a clear message.

---

## 5. Simulation core

### 5.1 Time

- Clock in **seconds** (float64) since the scenario's start timestamp (local wall-clock time of the data,
  `America/New_York`; DST is handled when building the calendar).
- Slots are aligned to clock hours. The old "day starts at 01:00" convention is dropped.
- Pure **discrete-event simulation**: the clock jumps from event to event. No fixed ticks.

### 5.2 Events

Single priority queue ordered by `(time, priority, sequence)` — the sequence number makes ties deterministic.

| Event | Emitted by | Effect |
|-------|-----------|--------|
| `SlotStart` | engine | demand generator materialises the next hour of requests |
| `TripRequest` | demand | user model searches a bike; schedules `TripStart` (after walking) or records a failure |
| `TripStart` | user model | bike leaves; schedules `TripEnd` |
| `TripEnd` | user model | bike parked (dockless) or docked (docked mode; may redirect if full) |
| `TruckArrive` | operator | truck reaches the target of its current command |
| `ServiceDone` | operator | pick-up / drop / swap completed; next command starts |
| `TruckIdle` | operator | queue empty → **notification to the controller** |
| `Wakeup` | controller | timer requested by a controller (e.g. "decide every 15 min") |

### 5.3 State (structure of arrays)

```
bikes:    node[int32], status[uint8] (PARKED | RIDING | ON_TRUCK | DEPOT | EXTERNAL),
          kind[uint8] (MECHANICAL | ELECTRIC), battery_km[float32]
nodes:    n_bikes[int32], per-node bike list (intrusive linked list in arrays: head/next/prev)
stations: (docked mode) capacity[int32], n_docked[int32]
trucks:   node, load list, battery stock, command queue, current command, busy_until
```

- Bike sets per node are O(1) to add/remove (array-based linked lists) — no Python dicts on hot paths.
- **Fleet conservation** is an invariant checked in debug mode after every event:
  `parked + riding + on_truck + depot + external == fleet_size`.

### 5.4 Partitions (aggregates for controllers)

Controllers register partitions of the nodes (`node -> group`), e.g. a 300 m grid.
The engine keeps per-group counters (bikes, e-bikes by battery band, recent failures, requests)
**incrementally** on every event, so reading them is O(groups), not O(nodes) or O(bikes).

### 5.5 Public API (sketch)

```python
sim = Simulator.from_scenario("scenarios/cambridge-2024", config)      # SimConfig, §10
notes = sim.reset(seed=7, start="2024-09-02T06:00")                    # deterministic

sim.add_partition("grid300", grid.node_to_zone)                        # optional views
sim.submit(truck=0, commands=[GoTo(node=1234), PickUp(5), GoTo(node=98), Drop(5)])

notes = sim.run(until=sim.any_truck_idle | sim.time_limit(3600))      # returns notifications
obs   = sim.observe()                       # read-only views: clock, bikes per node/group, trucks
fc    = sim.forecast(window_s=3600, partition="grid300")               # information layer, §8
kpis  = sim.kpis()                          # served, failed (by reason), walk m, truck km, ops
```

Commands return a `CommandResult` (requested vs. executed quantity, reason if partial/invalid)
delivered with the notifications.

---

## 6. Demand

### 6.1 Estimation (`bss_scenario`, pluggable `RateEstimator`)

Demand is factorised into a **shape** and a **volume**:

```
rate(o, d, weekday, hour, date) = volume(date) × shape(o, d | weekday, hour)
```

Evidence (2024, Cambridge): daily volume varies 3.3× across months (2,643 trips/day in Jan,
8,790 in Sep) and ±16 % day to day within a month; the hourly profile is almost identical all
year; the spatial pattern is stable from April to November and drifts in winter.

Default estimator `PooledShapeEstimator`:
1. **Fit period**: months where the shape is stable (default Apr–Nov 2024), configurable.
2. **Counts** per (origin station, destination station, weekday, hour), including an
   `EXTERNAL` pseudo-station for trips with one end outside Cambridge.
3. **Exposure** = number of *calendar* days of that weekday in the fit period, each weighted by its
   volume factor (fixes the old bug that divided by "days with at least one trip").
4. **Shrinkage** for sparse OD pairs (empirical Bayes): an OD rate is pulled towards
   `origin_rate(o, w, h) × p(d | o, coarser time band)`, so pairs seen once are not taken at face value.
5. **Volume**: `volume(date) = month_level(month) × day_factor`, with `day_factor` log-normal,
   fitted on the observed daily totals. Weather can later become a covariate of `day_factor`.

Later estimators (same interface): censoring-corrected demand (trips lost at empty stations),
weather-conditioned volume.

### 6.2 Spatial spreading (pluggable `Spreader`) — the old "interpolation"

Maps station-level demand to bike-network nodes for dockless operation.

- **Default `KernelSpreader`**: each station's demand is distributed over the nodes of its
  catchment area (Voronoi cell of the stations, truncated at a radius) with a distance kernel.
  Weights are normalised **per station**, so total demand is preserved.
- **`LegacyIDWSpreader`**: the thesis method, for comparison. Known flaws (documented, not fixed in
  legacy mode): it replicates rates into every empty OD entry, inflating demand where OSM nodes
  are dense; logged radius 500 m but used 250 m.
- **`StationSpreader`**: identity (docked mode).

### 6.3 Generators (pluggable `DemandGenerator`)

Both produce `TripRequest(time, origin_node, destination_node, ride_time_s, preferred_kind)`,
one hour at a time, on the `SlotStart` event.

- **`PoissonDemand`** (probabilistic): per date sample `day_factor`; per hour
  `N ~ Poisson(volume × Σ shape × 3600)`; OD pairs drawn with an alias table over the sparse shape;
  times uniform within the hour; nodes through the spreader; ride time from the fitted
  speed/duration model (§6.4). Fully vectorised: a whole hour is a handful of NumPy calls.
- **`ReplayDemand`** (recorded): requests from the cleaned 2024 trips, real times, real ride
  durations; nodes through the spreader (seeded jitter for dockless). Recorded trips are treated as
  **requests, not outcomes**: the simulator decides whether each one is served given the simulated
  state. Recorded bike ids are ignored. Known biases (documented): censored demand, displaced
  origins, no induced demand.

Typical use: train on `PoissonDemand`, evaluate on `ReplayDemand` for held-out dates, validate the
simulator by replaying history (§11).

### 6.4 Ride time

Fitted from data instead of the old fixed truncated Gaussian (15 ± 5 km/h): speed distribution
per distance band (from network distance between stations / recorded duration, excluding round
trips), separately for e-bikes and classic bikes. Round trips (o = d) use the empirical duration
distribution.

### 6.5 Trips with one end outside the city

`EXTERNAL` replaces the old virtual station `10000` and the pool of 1000 "outside bikes":
- Departure to `EXTERNAL`: the bike leaves the system (status `EXTERNAL`).
- Arrival from `EXTERNAL`: a bike from the external reservoir appears at the destination node.
- Reservoir size and battery distribution are configuration; fleet conservation includes it.

---

## 7. Users and fleet

### 7.1 User model (pluggable `UserModel`)

On `TripRequest` at origin node `o`:
1. Candidate bikes: nodes within the walking radius (precomputed CSR, sorted by distance),
   with a usable bike (e-bike battery ≥ trip distance + reserve, or mechanical).
2. The user considers candidates by walking distance `w` and accepts with probability
   `p(w) = exp(-w / β)` (default β = 150 m, radius 400 m; configurable, literature-based since walking
   is not observable in the data). Optional preference for e-bikes.
3. Accepted → `TripStart` after `w / walking_speed` (1.4 m/s). Otherwise → failure with reason
   `NO_BIKE_IN_RANGE`, `DECLINED_WALK` or `LOW_BATTERY_ONLY`.
4. Docked mode: candidates are stations with available bikes; on `TripEnd` at a full station the
   user rides to the nearest station with a free dock (extra time and distance), failure
   `NO_DOCK` if none within range.

### 7.2 Fleet and energy

```yaml
fleet:
  size: 1500
  ebike_share: 0.2          # 0 = all mechanical, 1 = all electric
  battery_range_km: 60      # e-bike full range
  initial_battery: {dist: uniform, low: 0.3, high: 1.0}
  initial_distribution: proportional_to_demand   # | uniform | from_snapshot
energy:
  consumption_km_per_km: 1.0
  dead_ebike_rideable: false
```

Riding consumes `distance × consumption`. No charging except battery swaps by operators
(stations with chargers can be added later).

---

## 8. Information layer (what controllers may see)

Controllers never access the event queue. They get:

- **Observation**: current state views (bikes per node/group, battery bands, trucks, clock, KPIs).
- **Forecast** from a configured `ForecastProvider`:

| Provider | Content | Use |
|----------|---------|-----|
| `MeanForecast` | expected departures/arrivals per group from the demand model (analytic) | low variance |
| `SampledForecast` | one or K scenarios sampled from the demand model with an independent RNG | current critic |
| `OracleForecast` | the simulator's realised future requests | **ablation only**; requires `allow_oracle=True`, logged in run metadata |

- **In-flight trips**: `inflight_knowledge: none | count | destination`.
  `destination` (arrival time and place of trips in progress) reproduces the current critic.
  **Note for review:** a real dockless operator usually knows a bike is in use but not where it
  will end. `count` is the realistic option; `destination` is kept to match the current setup.

---

## 9. Operators (trucks)

```yaml
trucks:
  count: 1
  capacity: 30
  depot: {lat: ..., lon: ...}
  start: depot              # | random_node | node id
  initial_load: 15
  battery_stock: 20         # charged batteries carried, for swaps
  service_time_s: {pickup_per_bike: 30, drop_per_bike: 30, swap_per_bike: 60, stop_overhead: 60}
```

Commands (queued per truck, executed in order):

| Command | Semantics |
|---------|-----------|
| `GoTo(node)` | drive on the drive network (time-dependent travel time) to the drive node nearest to `node` |
| `PickUp(n, policy="lowest_battery" \| "nearest", radius_m=100)` | collect up to `n` bikes within `radius_m` of the stop; bikes leave the street when the service is **done** |
| `Drop(n, at=None \| [nodes], spread="here" \| "nodes")` | place up to `n` bikes at the stop or at given nodes within walking reach; available when the service is done |
| `SwapBattery(n)` | replace the batteries of up to `n` lowest-charge e-bikes nearby |
| `ToDepot()` / `Restock(load=…, batteries=…)` | go to the depot, unload/load bikes and batteries |
| `Wait(seconds)` | stay idle-but-busy for a duration |

**Travel time** (`TravelModel`, issue #28; data in #17):
1. free-flow speeds from OSM road attributes;
2. city-wide congestion factor per weekday-hour (TomTom Traffic Index, Boston);
3. per-segment historical speeds per weekday-hour (TomTom Traffic Stats), map-matched to OSM drive edges.
Hours are grouped into a few traffic regimes so that time matrices stay small (one ~16 MB matrix per regime).

**Energy** (`VehicleEnergyModel`, issue #28): consumption as a function of speed (constant-speed consumption
curve, as used by TomTom's EV routing), stops and load. The truck battery (or fuel) is a resource;
commands that would exhaust it are rejected, and `ToDepot()` recharges.

Execution is honest about time: driving, stopping and handling all take time, and the state
changes when they complete (fixes the old asymmetry where pick-ups happened at decision time).
Partial execution (e.g. fewer bikes available) is reported in the `CommandResult`.

---

## 10. Configuration and reproducibility

- Configuration: typed, frozen dataclasses loadable from YAML (`SimConfig`, `DemandConfig`,
  `FleetConfig`, `TruckConfig`, `UserConfig`, `InfoConfig`), validated at load, serialised with every
  run (with its hash).
- RNG: one `numpy.random.SeedSequence(seed)` per run, **spawned** into independent streams per
  component (demand, users, travel noise, fleet init, forecast). Adding a component never changes
  the streams of the others.
- No process-global state (no class-level counters), no background processes, no disk caches on
  the critical path.
- `reset(seed)` is a pure function of `(scenario, config, seed, start)`: tested by running a seed,
  then other seeds, then the first seed again and comparing event logs byte for byte.

---

## 11. Telemetry and validation

**Telemetry**: an optional event log (Arrow → Parquet) with trip outcomes (served/failed + reason,
walk distance, wait), truck operations and periodic snapshots. Disabled → near-zero cost. KPIs are
always available in memory. The results web app reads the logs.

**Validation plan**

1. Unit tests and invariants (fleet conservation, non-negative counts, capacity limits).
2. Determinism tests (§10).
3. Statistical checks of the generators against data: requests per hour, OD distribution, ride
   durations (JSD / KS against held-out months).
4. **Calibration by replay**: replay held-out 2024 months and compare served trips, durations and
   spatial patterns to the records.
5. Equivalence with the old simulator where semantics overlap (old data, old settings) — to know
   which differences come from the model changes.

---

## 12. Performance budget

Measured prototype (pure Python, heap queue, array state): one week at Cambridge volume
(~66 k trips, ~80 k events) in **0.12 s** (1.5 µs/event) plus 5 ms of vectorised sampling.

Budget, single core, Python 3.11, enforced by benchmark tests:

| Scenario | Budget |
|----------|--------|
| Load the Cambridge scenario | ≤ 2 s |
| 1 simulated week, no trucks, logging off | ≤ 0.5 s |
| 1 simulated week, 1 truck with random commands every ~5 min | ≤ 1 s |
| Peak memory | ≤ 600 MB |

Benchmarks run on Python 3.11, 3.14 and free-threaded 3.14t (parallel simulators in threads),
and later against the Rust engine (§13). Training-side speed comes from: decisions only when a
truck is idle, many simulators in parallel with batched policy inference, and no per-step
recomputation of global quantities.

---

## 13. Rust engine (later)

The engine boundary — array state, event queue, typed commands, notifications — is designed to be
reimplemented in Rust (PyO3 + maturin) behind the same Python API. Models that are already
vectorised (demand sampling) can stay in NumPy. The two engines must produce identical event logs
for the same seed (where the same RNG streams are used) or pass statistical equivalence tests.

---

## 14. Adapters

**`bss_gym.GridZoneEnv`** reproduces the current environment on top of `bss_sim`:
- 300 m square grid as a partition; actions STAY / UP / DOWN / LEFT / RIGHT / DROP / PICK_UP / CHARGE
  translated into commands (`GoTo(zone centre)`, `Drop(1)`, `PickUp(1)`, `SwapBattery(1)`, `Wait(60)`).
- Observation, critic score and reward ported as **separate components** so they can be swapped.
- Semi-MDP stepping: one decision per `TruckIdle`; `info["elapsed_s"]` replaces the old "steps".
- Standard Gymnasium semantics (`terminated`, `truncated`), proper observation spaces.

**`bss_policies`**: `StaticTSP` (scheduled rebalancing with real truck travel and service times —
no more instantaneous teleport), `DoNothing`, `GreedyDeficit`.

---

## 15. Milestones

| M | Deliverable | Done when |
|---|-------------|-----------|
| M1 | `bss_scenario`: trip cleaning (2024 schema), networks, stations, distance/time matrices | bundle builds for Cambridge 2024; schema validated |
| M2 | Demand estimation + spreader + both generators | statistical checks vs held-out months pass |
| M3 | `bss_sim` core: engine, state, users, fleet, energy, telemetry | invariants + determinism + week ≤ 0.5 s |
| M4 | Trucks, commands, information layer | command semantics tests; week with truck ≤ 1 s |
| M5 | `bss_policies` baselines | static TSP and greedy run end to end |
| M6 | `bss_gym.GridZoneEnv` + existing DQN trains on it | training runs; comparison with old env documented |
| M7 | Calibration report (replay vs records) | report published |
| M8 | Rust engine prototype + benchmark (3.11 / 3.14 / 3.14t / Rust) | numbers in a report; decision |

---

## 16. Open questions for review

1. **In-flight knowledge** (§8): default `count` (realistic) or `destination` (as in the current critic)?
2. **Fit period**: pool Apr–Nov 2024 for the shape, or Sep–Oct only?
3. **Weekday resolution**: shape per weekday (7 × 24 slots) or day types (Mon–Thu, Fri, Sat, Sun)?
   Per weekday is the default if the pooled data supports it.
4. **Walking model defaults**: β = 150 m, radius 400 m, 1.4 m/s — acceptable as a starting point?
5. **External trips**: reservoir model as in §6.5, or keep external trips out of the simulation entirely?
6. **Depot location and fleet size** for Cambridge 2024 (the old value 18 was a cell id).
7. **Truck battery stock / swap semantics**: is swapping batteries on the street the operation you
   want to model, or should the truck collect e-bikes to charge them at the depot?
8. **Truck vehicle**: electric van (model / battery capacity / consumption curve) or diesel? Determines the energy model defaults.
