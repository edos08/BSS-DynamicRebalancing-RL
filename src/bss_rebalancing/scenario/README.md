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

## Status

Milestone M1 (epic #38). Commands are added as they are implemented.
