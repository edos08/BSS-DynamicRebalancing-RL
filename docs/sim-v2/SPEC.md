# BSS Simulator v2 — Specification

Status: **accepted** (review decisions in §16) · Date: 2026-11-25

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
| D15 | Controllers do **not** know where trips in progress will end. Expected arrivals of in-flight bikes are a forecast, and every forecast must pass calibration tests (§8.1). |
| D16 | Demand shape fitted on **Apr–Nov 2024** pooled; monthly volume level per month. |
| D17 | Walking model defaults β = 150 m, radius 400 m, 1.4 m/s. |
| D18 | Trips with one end outside Cambridge use the **external reservoir** (§6.5). |
| D19 | Depot near **First Street × Cambridge Street** (East Cambridge); default fleet size from the operator's live feed (§7.2). |
| D20 | Trucks are **electric vans** that **swap e-bike batteries on the street**; the battery stock is limited, and an empty stock sends the truck to the depot, after which it **resumes** the interrupted command (§9.1). The van itself is assumed **always charged**: energy is reported, not constrained. |

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

A scenario is a directory produced by `bss-scenario build <config.yaml> --out <dir>`
(config example: `configs/scenarios/cambridge-2024.yaml`); the simulator only reads it.
`bss-scenario check <dir>` validates it.

```
data/scenarios/cambridge-2024/
├── scenario.yaml              # id, schema_version, timezone, slot length, fit period, components, projection
├── manifest.json              # inputs and outputs with SHA-256, package versions, command line, config
├── trips/
│   ├── 2024-MM.parquet        # cleaned recorded trips, one file per local start month (flags, city ends)
│   └── cleaning_report.json
├── network/
│   ├── bike_nodes.parquet     # node_idx, osm_id, lat, lon, x, y (metres, local projection shared by both networks)
│   ├── bike_edges.parquet     # u, v, length_m, highway, maxspeed, oneway, name
│   ├── bike_dist.npy          # float32 [N, N] directed shortest paths on the bike network (m)
│   ├── bike_walk_nbrs.npz     # CSR walking neighbours (undirected) up to walk_radius_max_m, sorted by distance
│   ├── drive_nodes.parquet    # drive network without busways
│   ├── drive_edges.parquet    # + speed_kmh, speed_source, freeflow_s
│   ├── drive_time.npy         # float32 [M, M] free-flow driving time (s), one-way streets respected
│   ├── bike_to_drive.npz      # node: nearest drive node of each bike node; dist_m
│   └── network_report.json
├── stations/
│   ├── stations.parquet       # station_id, name, lat, lon, in_city, usage, capacity (+source), bike_node
│   ├── stations.npz           # in-city stations as arrays for the simulator (id, bike_node, capacity, lat, lon)
│   └── stations_report.json
├── traffic/                   # (#57) congestion factors per day type x hour x road class
└── demand/                    # (M2) shape, volume, spreading, durations
```

Sizes for Cambridge 2024 (bike network 4,896 nodes, drive network 1,942 nodes, 91 stations inside
the city out of 512 on trips touching it): 182 MB, built in 15 s.
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
2. **Counts** per (origin station, destination station, day group, hour), including an
   `EXTERNAL` pseudo-station for trips with one end outside Cambridge.
3. **Exposure** = number of *calendar* days of that weekday in the fit period, each weighted by its
   volume factor (fixes the old bug that divided by "days with at least one trip").
4. **Shrinkage** for sparse OD pairs (empirical Bayes): an OD rate is pulled towards
   `origin_rate(o, w, h) × p(d | o, coarser time band)`, so pairs seen once are not taken at face value.
5. **Volume**: `volume(date) = month_level(month) × day_factor`, with `day_factor` log-normal,
   fitted on the observed daily totals. Weather can later become a covariate of `day_factor`.
6. **Day grouping** (`day_groups`, configurable): which days share one hourly shape.
   `per_weekday` = 7 separate shapes (Mon, Tue, …, Sun); `day_types` = 4 shapes
   (Mon–Thu, Fri, Sat, Sun); `workday_weekend` = 2. More groups capture more detail but each has
   less data. The default is chosen in M2 by held-out log-likelihood on Apr–Nov 2024 (#18).

**Selection result (Cambridge, Apr–Nov 2024, #80).** 20 % of days held out (stratified by weekday);
score = mean conditional log-likelihood per held-out trip of (hour, origin, destination) given the
day's total. Best: `day_types`, alpha = 1024, beta = 512 (−9.7405 per trip); `workday_weekend`
−0.0014 and `per_weekday` −0.0092 behind. Without smoothing the best scheme loses 0.98 per trip
and gives zero probability to 24,511 held-out trips: smoothing matters far more than the grouping.
Holidays are put in the Sunday group.

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

**Validation (#85).** `scripts/validate_demand.py` refits the model on the training days of the
selection split and compares `PoissonDemand` with the requests recorded on the 49 held-out days,
against an empirical baseline (the training days' own recorded requests). Report:
[`reports/demand-validation.md`](reports/demand-validation.md). Cambridge 2024: daily volume within
1–7 %; hourly, OD and origin distributions as close as the empirical baseline; ride times KS
0.02–0.05, because generated rides use node-to-node distances after spreading while recorded
durations are between stations.

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

Implementation (#100): a trip to `EXTERNAL` leaves at `TripStart`, after the user has walked to the
bike and its battery is charged for `energy.external_trip_km`. A request from `EXTERNAL` takes a
bike of the preferred kind from the reservoir if it has one, otherwise one of the other kind, or fails
with `EXTERNAL_EMPTY` when the reservoir is empty. The bike arrives at its destination after the
request's ride time, or `external.ride_time_s` (15 min) when that is unknown. An e-bike gets a fresh
battery drawn from `fleet.initial_battery`, because bikes are charged outside the city too.

**Drift and the default size.** Recorded trips are outcomes, but the simulator treats them as
requests. Requests from inside the city can fail (no bike nearby, walk declined), while requests
from outside find a bike as long as the reservoir has one. So the city gains bikes even though the
fitted demand has slightly more trips out than in (in/out = 0.97). One simulated week of September
2024 (dockless, 800 bikes, no trucks) shows it:

| reservoir | requests from outside served | trips out of the city | city bikes after 7 days | `EXTERNAL_EMPTY` |
|---|---|---|---|---|
| 500 | 13,876 | 13,376 | 1,288 | 6.3 % of requests |
| 4,000 (all bikes outside Cambridge) | 17,765 | 15,125 | 3,421 | 0 |

A finite reservoir limits the drift: once it is empty, a bike can come in only after another has
gone out. The default is **1,000**, the value of the old simulator's pool of outside bikes.
Calibration in M7 (#32) checks the reservoir together with the fleet size, against replayed 2024
days. KPIs give the service level both for all requests and for requests starting in the city,
which are the ones a controller can influence.

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
  size: 800                 # see "Default fleet size" below
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

**Default fleet size.** BlueBikes' public GBFS feed (snapshot 2026-11-25) lists, for the
Cambridge region, 101 stations with 2,011 docks and 736 bikes parked (234 e-bikes); the whole
system has about 4,900 bikes parked across 605 stations. Adding bikes in use, about 800 bikes
serve Cambridge, roughly 40 % of its docks. The default is 800 with `ebike_share` taken from the
2024 data; the M7 calibration (#32) tunes it so that replayed 2024 days reproduce the recorded
service level.

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

- **In-flight trips**: `inflight_knowledge: none | count | origin | destination`.

| Level | Controller sees | Realistic? |
|-------|-----------------|------------|
| `none` | nothing about trips in progress | pessimistic |
| `count` | number of bikes currently in use | yes |
| `origin` (**default**) | for each trip in progress: where and when it started (the operator unlocked the bike) | yes |
| `destination` | where and when each trip will end (the current critic) | **no**: ablation only, needs `allow_oracle=True` |

  With `count` and `origin` the controller does not know where in-flight bikes will arrive, so the
  forecast predicts it (§8.1).

### 8.1 Arrival forecast for trips in progress

For a trip that started at origin `o` at time `s` and has been running for `a = now − s`, the
probability that it ends in destination group `g` within the window `(now, now + h]` is

```
                       Σ_{d ∈ g} p(d | o, s) · [F_od(a + h) − F_od(a)]
P(arrive in g, by now+h) = ------------------------------------------------
                       Σ_{d}     p(d | o, s) · [1 − F_od(a)]
```

where `p(d | o, s)` comes from the fitted demand shape (day group and hour of `s`) and `F_od` is
the ride-time distribution (§6.4). The denominator conditions on the trip still running, so a
trip that has already lasted long is more likely heading far.
With `count`, `o` and `s` are replaced by their distribution given the recent departures.
Expected arrivals per group are the sum over trips in progress; the sampled forecast draws each
trip's destination and arrival time from the same distribution (an independent RNG stream).
Future departures in the window are added from the demand model as before.

**Calibration is part of the contract.** A wrong forecast gives the agent wrong information, so
each forecast provider ships with tests (#29) run on simulated and on held-out 2024 data:

- **Bias**: mean (predicted − realised) arrivals per group and horizon ≈ 0 (within the Monte
  Carlo error).
- **Coverage**: realised counts fall inside the 50 % / 90 % predictive intervals about 50 % / 90 %
  of the time (PIT histogram flat).
- **Sharpness** reported next to `destination` (oracle) and `none`, so the cost of not knowing
  destinations is measured, not assumed.

---

## 9. Operators (trucks)

```yaml
trucks:
  count: 1
  capacity: 30
  depot: {lat: 42.37065, lon: -71.07692}   # First St × Cambridge St (OSM drive node 8508760490)
  start: depot              # | random_node | node id
  initial_load: 15
  battery_stock:
    max: 60                 # charged e-bike batteries the truck can carry
    initial: 60
    on_empty: resume        # | idle  (see §9.1)
  service_time_s: {pickup_per_bike: 30, drop_per_bike: 30, swap_per_bike: 60, stop_overhead: 60,
                   depot_restock: 600}
  vehicle: e_transit        # electric van preset, see "Energy"
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

**Travel time** (`TravelModel`, issue #28; data in #17). Pluggable sources, in increasing detail:

1. `FreeFlow`: speeds from OSM road attributes (speed limit / road class), no traffic.
2. **`SampledCongestion` (default)**: a congestion model fitted on sampled routing queries.
   - Sample ~200 origin–destination pairs over the drive network, stratified by distance and
     area so that all road classes and parts of the city are covered.
   - Query a routing API with historical traffic (TomTom Routing API `departAt`, free tier of
     2,500 requests/day) for every hour of a typical weekday, Saturday and Sunday
     (~200 × 24 × 3 ≈ 14k queries, run over several days). Raw responses are cached on disk so the
     collection is resumable and the fit is reproducible without new queries.
   - Fit a multiplicative slowdown factor `f(day type, hour, road class)` on top of free-flow
     edge times (least squares on log travel time ratios, with smoothing across hours).
   - Hold out 20 % of the pairs to report the error of the fitted model; cross-check a subset
     against the Google Routes API (traffic-aware, free monthly allowance) as an independent source.
   - The fitted table (a few hundred numbers) is stored in the scenario bundle with its query
     dates and source; API keys come from environment variables.
3. `CityIndex`: one city-wide factor per weekday-hour (TomTom Traffic Index, Boston), as a fallback.
4. `SegmentSpeeds`: per-segment historical speeds (TomTom Traffic Stats, paid / trial),
   map-matched to OSM drive edges, if access becomes available.

Hours with similar factors are grouped into a few traffic regimes so that time matrices stay small
(one ~16 MB matrix per regime).

**Energy** (`VehicleEnergyModel`, issue #28): trucks are **electric vans**. Consumption is a
function of speed (constant-speed consumption curve, as used by TomTom's EV routing), stops and
load. The default preset `e_transit` follows a Ford E-Transit cargo van (≈ 68 kWh usable battery,
≈ 0.3–0.4 kWh/km in urban driving); the exact curve and figures are fixed and sourced in #28.

The van battery is **not a constraint**: vans are assumed always charged, because a van running
low is swapped for a charged one at its next depot visit. This is negligible in practice: a shift
in Cambridge (≈ 18 km²) drives on the order of 60–120 km, i.e. roughly 20–50 kWh, below the usable
capacity. Energy is a **KPI**: kWh per truck and per shift (and derived cost/emissions) are
reported, and a run is flagged if a truck's kWh between two depot visits exceeds the usable
capacity, so the assumption is checked rather than trusted.

**Cost and emissions** (KPIs only, for later analyses; they never change the dynamics):

```yaml
costs:
  electricity_eur_per_kwh: ...      # energy cost of the vans
  driver_eur_per_hour: ...          # labour, per truck-hour on shift
  van_eur_per_km: ...               # maintenance / depreciation
  battery_swap_eur: ...             # handling cost per swapped battery
emissions:
  grid_kgco2e_per_kwh: ...          # electricity carbon intensity (ISO New England average)
```

Reported per truck, per shift and per run: kWh, km driven, operating cost, kg CO2e, and per
rebalanced bike / per extra served trip. Default values are chosen and sourced in #28.

### 9.1 Depot returns and resuming

Battery swaps happen **on the street**: `SwapBattery(n)` replaces the batteries of up to `n`
e-bikes at the stop, using charged batteries from the truck's stock. The stock is limited
(`battery_stock.max`); used batteries stay on the truck until the depot.

When the stock is empty and the current or next command needs it:

- `on_empty: resume` (**default**): the truck finishes the service in progress with what it has,
  then inserts `ToDepot()` + `Restock()` at the head of its queue, drives to the depot, swaps
  the used batteries for charged ones (`depot_restock` time), and then **resumes the interrupted
  command where it was heading** (e.g. the `GoTo` target), followed by the rest of the queue. The
  controller is notified (`TruckRestocking`, then `TruckResumed`) but does not have to act.
- `on_empty: idle`: the truck stops, reports `TruckIdle(reason=STOCK_EMPTY)` and waits for the
  controller to decide, e.g. to send it to the depot itself.

Resuming the interrupted command, rather than returning to the point where the stock ran out,
keeps the controller's plan valid and avoids a wasted trip back.

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
always available in memory: service level, failures by reason, walk distances, bikes moved,
batteries swapped, and per truck km, driving/service/idle time, kWh, operating cost and CO2e (§9).
The results web app reads the logs.

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

## 16. Review decisions

Answers to the review questions (2026-11-25), recorded as D15–D20 in §2.

| # | Question | Decision |
|---|----------|----------|
| 1 | In-flight knowledge | Destinations of trips in progress are **not** known. Default `origin` (where and when trips started), `count` also realistic; arrivals are forecast with calibrated probabilities (§8.1). `destination` is an oracle ablation. |
| 2 | Fit period | Apr–Nov 2024 pooled. |
| 3 | Weekday resolution | Configurable `day_groups` (§6.1); default chosen by held-out likelihood in M2. |
| 4 | Walking model | β = 150 m, radius 400 m, 1.4 m/s. |
| 5 | External trips | External reservoir (§6.5). |
| 6 | Depot and fleet | Depot at First St × Cambridge St; fleet 800 by default (GBFS, §7.2), tuned in M7. |
| 7 | Battery swaps | On the street, limited stock, depot return then resume (§9.1). |
| 8 | Truck vehicle | Electric van, `e_transit` preset (§9); always charged, energy reported as a KPI (#50). |

Still open (not blocking M1): TomTom data access (Traffic Stats trial vs Routing API with
historical traffic, #17).
