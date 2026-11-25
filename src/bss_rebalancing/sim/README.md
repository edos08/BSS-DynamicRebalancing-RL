# bss-sim

The BSS simulator v2 (`docs/sim-v2/SPEC.md`): a discrete-event simulator of a bike-sharing city,
its users, fleet and operators. It knows nothing about reinforcement learning, depends only on
NumPy, and reads scenario bundles built by `bss-scenario` (never importing it).

## Status

Milestone M2: demand generators (`bss_sim.demand`). The engine comes in M3 (epic #40).

```python
import numpy as np
from bss_sim.demand import DemandModel, PoissonDemand

model = DemandModel.load("data/scenarios/cambridge-2024")
gen = PoissonDemand(model, seed=1)
requests = gen.hour(datetime.date(2024, 9, 17), 8)   # TripRequests for 08:00-09:00
```
