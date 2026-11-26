# bss-sim

The BSS simulator v2 (`docs/sim-v2/SPEC.md`): a discrete-event simulator of a bike-sharing city,
its users, fleet and operators. It knows nothing about reinforcement learning, depends only on
NumPy, and reads scenario bundles built by `bss-scenario` (never importing it).

## Status

- M2: demand generators.
- M3 (epic #40): simulation core: engine, bike state, batteries, users, external reservoir,
  partitions, telemetry, benchmarks.
- M4 (epic #41): trucks and their commands, travel times, van energy/cost/emissions, observation,
  in-flight knowledge, calibrated forecasts.
- Next: baseline policies (M5), Gym adapter (M6).

```python
from bss_sim.commands import Drop, GoTo, PickUp, SwapBattery
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator

sim = Simulator.from_scenario("data/scenarios/cambridge-2024", SimConfig())   # dockless, 800 bikes, 1 van
sim.add_partition("grid", node_to_zone)               # your zones: one group id per bike node
notes = sim.reset(seed=7, start="2024-09-16T06:00")   # deterministic in (scenario, config, seed, start)
while sim.now < 22 * 3600:
    obs = sim.observe()                               # bikes, zones, trucks, trips in progress, KPIs
    fc = sim.forecast(1800, "grid")                   # expected departures / arrivals per zone
    sim.submit(0, [GoTo(1234), PickUp(5), GoTo(98), Drop(5), SwapBattery(3)])
    notes = sim.run(until=OnNotification("truck_idle") | TimeLimit(3600))
print(sim.kpis())   # service, failures by reason, walking, truck km/kWh/cost/CO2e, bikes moved
```

## Modules

| Module | Content |
|---|---|
| `config` | `SimConfig` and its sections (fleet, energy, users, external, trucks, travel, vehicle, costs, info, telemetry) |
| `rng` | named random streams derived from the run seed |
| `engine` | event queue, stop conditions (`TimeLimit`, `At`, `OnNotification`), notifications |
| `scenario` | what is read from a bundle (npz/json/npy only) |
| `demand` | `PoissonDemand`, `ReplayDemand`, ride times, spreader |
| `state` | bike arrays, per-node lists, external reservoir, invariant check |
| `energy` | e-bike battery: consumption, usability, bands |
| `users` | `WalkingUsers`: nearest usable bike within the walking radius, acceptance `exp(-w/β)` |
| `commands` | truck commands and `CommandResult` |
| `trucks` | truck state and command validation |
| `travel` | free-flow and congestion-factor travel times on the drive network |
| `vehicle` | electric van energy, operating cost, emissions (KPIs) |
| `partitions` | per-zone counters kept incrementally |
| `info` | `Observation`, in-flight knowledge levels |
| `forecast` | mean, sampled and oracle forecasts per zone |
| `forecast_check` | calibration of the forecasts against what happens |
| `telemetry` | event log, snapshots, metadata |
| `simulator` | the event loop tying it together |
| `synthetic`, `bench` | Cambridge-size synthetic bundle; benchmarks and budget |
| `checks` | statistical comparison of request streams |

Reports: `docs/sim-v2/reports/` (demand validation, performance, forecast calibration).
