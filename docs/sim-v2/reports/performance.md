# Simulator performance (M3)

Budget from spec §12. The `Benchmark` CI job enforces it on every PR, on the synthetic
Cambridge-size bundle (`python -m bss_sim.bench --make-synthetic DIR`, issue #104), because CI has
no real data.

**Setup.** Default configuration: dockless, 800 bikes, external reservoir 1,000, logging off,
Poisson demand. One week runs from Monday 2024-09-16 00:00, about 65k requests. Times are the best
of 3 runs on a single core.

## GitHub runners (ubuntu-latest), synthetic bundle

Run of PR #116:

| Python | NumPy | load | **week** | µs / request | docked week | week + 500-group partition | peak memory | 4 threads vs serial |
|---|---|---|---|---|---|---|---|---|
| 3.11.18 | 2.4.6 | 0.06 s | **0.27 s** | 4.2 | 0.22 s | 0.30 s | 219 MB | 0.85× |
| 3.14.10 | 2.5.5 | 0.06 s | **0.32 s** | 4.9 | 0.26 s | 0.35 s | 219 MB | 0.82× |
| 3.14.10 free-threaded | 2.5.5 | 0.06 s | **0.26 s** | 4.1 | 0.23 s | 0.34 s | 235 MB | **2.61×** |
| *budget* | | *2 s* | *0.5 s* | | | | *600 MB* | |

Free-threaded Python runs independent simulators in threads 2.6× faster than one after another
(4 threads on the runner's 4 vCPUs). On the GIL builds, threads are slightly slower than serial.
This supports training with many simulators in one process.
Python 3.14 with the GIL is about 15 % slower than 3.11 on this workload.

## Real Cambridge 2024 bundle (this server, shared and loaded)

| | |
|---|---|
| load | 0.08 s |
| one week (64,864 requests) | 0.35 s (5.4 µs per request) |
| docked week | 0.34 s |
| week with a 500-group partition | 0.38 s |
| peak memory | 202 MB |

## What was optimised (#105)

The first complete loop took 0.80 s per week on Cambridge. These changes brought it to 0.35 s
without changing any outcome. Event logs before and after are byte-identical: 3 days of
dockless and docked runs with a partition, 72k and 95k events.

| Change | Why |
|---|---|
| Requests are read in time order from the hour's batch, not pushed on the heap | one heap push and pop less per request |
| Requests are tuples of Python scalars, with the in-city distance computed for the whole hour in one NumPy step | no NumPy scalar reads per request |
| Fleet state lives in `array.array` buffers, with NumPy views of the same memory | scalar updates 5× cheaper than on NumPy; vectorised code unchanged |
| Per-node bike lists and per-bike trip data are Python lists | element access on the hot path |
| The user model scans the origin node first; the vectorised neighbourhood scan only runs if needed | most users find a bike where they are |
| No `TripStart` event when the user does not walk | same order as an event at the same time |
| Ride times are sampled from one quantile table for all requests | 16 masked `np.interp` calls → one gather |
| Bike distances are read through a plain ndarray view of the memory map | `memmap.__getitem__` is Python code |

## Not yet measured

The truck benchmark (1 truck with random commands every ~5 min, budget 1 s) comes with M4. The
Rust engine comparison comes with M8.

Reproduce: `python -m bss_sim.bench <bundle> --json out.json`.
