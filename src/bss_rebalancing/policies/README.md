# bss-policies

Baseline controllers for the BSS simulator v2 (`docs/sim-v2/SPEC.md` §14), used as references for
the RL agents. They drive `bss_sim` only through its public API; `bss_sim` never imports them.

| Controller | What it does |
|---|---|
| `DoNothing` | no rebalancing: trucks stay at the depot |
| `StaticTSP` | at scheduled hours, rebalances towards target bikes per zone along a capacitated route, driven for real (#152) |
| `GreedyDeficit` | whenever a truck is idle, serves the zone with the best expected gain per travel time, from the forecasts (#153) |

```python
from bss_policies import DoNothing, run_episode
from bss_sim.simulator import Simulator

sim = Simulator.from_scenario("data/scenarios/cambridge-2024")
res = run_episode(sim, DoNothing(), seed=0, start="2024-09-16T00:00", duration_s=7 * 86400)
print(res.kpis["city_service_level"], res.decisions, res.wall_s)
```
