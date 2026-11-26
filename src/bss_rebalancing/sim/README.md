# bss-sim

The BSS simulator v2 (`docs/sim-v2/SPEC.md`): a discrete-event simulator of a bike-sharing city,
its users, fleet and operators. It knows nothing about reinforcement learning, depends only on
NumPy, and reads scenario bundles built by `bss-scenario` (never importing it).

## Status

- M2: demand generators (`bss_sim.demand`).
- M3 (epic #40): simulation core — engine, bike state, batteries, users, external reservoir.
- Trucks and the information layer come in M4.

```python
from bss_sim.config import SimConfig
from bss_sim.engine import OnNotification, TimeLimit
from bss_sim.simulator import Simulator

sim = Simulator.from_scenario("data/scenarios/cambridge-2024", SimConfig())   # dockless, 800 bikes
sim.reset(seed=7, start="2024-09-16T06:00")          # deterministic in (scenario, config, seed, start)
sim.wakeup(after=900, tag="decide")                   # controller timer
notes = sim.run(until=OnNotification("wakeup") | TimeLimit(3600))
sim.run(until=TimeLimit(7 * 86400))
print(sim.kpis())   # requests, served, failures by reason, service level, walking, km ridden
```

## Modules

| Module | Content |
|---|---|
| `config` | `SimConfig` and its sections (fleet, energy, users, external, telemetry), YAML/dict loading, hash |
| `rng` | named random streams derived from the run seed |
| `engine` | event queue, stop conditions (`TimeLimit`, `At`, `OnNotification`), notifications |
| `scenario` | what is read from a bundle (npz/json/npy only) |
| `demand` | `PoissonDemand`, `ReplayDemand`, ride times, spreader |
| `state` | bike arrays, per-node lists, external reservoir, invariant check |
| `energy` | e-bike battery: consumption, usability, bands |
| `users` | `WalkingUsers`: nearest usable bike within the walking radius, acceptance `exp(-w/β)` |
| `simulator` | the event loop tying it together |
| `checks` | statistical comparison of request streams |
