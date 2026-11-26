# Baseline controllers (M5)

Reference controllers for the RL agents (spec §14, epic #42), on Cambridge 2024.

## Setup

- **Simulation**: one week from Mon 2024-09-16, 3 seeds. Demand from the fitted model (Poisson), 800
  bikes, one Ford E-Transit van starting at the depot with 15 bikes and 60 batteries. The van works
  06:00–22:00: at 22:00 it returns to the depot and restocks, and it is idle overnight. Travel at free
  flow; the traffic factors come with #57.
- **Dockless**: bikes start where demand is.
- **Docked**: 1,824 docks; bikes start spread by dock capacity.
- **Controllers** (`bss_policies`):
  - `DoNothing`: the van stays at the depot.
  - `StaticTSP`: plans every 3 h (or every hour) from 06:00. It computes target bikes per zone from the
    forecast and drives a capacitated pick-up and delivery route until the next plan.
  - `GreedyDeficit`: every time the van is idle, it re-plans the single best visit from a fresh
    observation and a 1 h forecast.
  - `+ swaps`: also swaps the batteries of e-bikes below 25 % at visited stops.
- **Zones**: a 300 m grid (231 zones) in dockless mode, station catchments (91) in docked mode.
- **Reproduce**: `python scripts/run_baselines.py data/scenarios/cambridge-2024 --days 7 --seeds 3`
  (episodes in `baselines.json`).

## Results (means over seeds; service level ± s.d.)

**dockless**

| controller | served (city) | failed requests | mean walk m | redirects | bikes moved | swaps | van km | kWh | cost USD | kg CO2e | decisions | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DoNothing | 78.1% ± 0.7% | 12,711 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 |
| StaticTSP 3 h | 80.4% ± 0.5% | 11,243 | 25 | 0 | 4,541 | 0 | 731 | 259 | 2,586 | 73 | 118 | 2.9 |
| StaticTSP 1 h | 80.0% ± 0.6% | 11,499 | 26 | 0 | 4,505 | 0 | 570 | 208 | 2,403 | 59 | 296 | 1.8 |
| GreedyDeficit | 80.3% ± 0.7% | 11,340 | 25 | 0 | 4,928 | 0 | 717 | 255 | 2,723 | 72 | 718 | 19.5 |
| GreedyDeficit + swaps | 80.4% ± 0.6% | 11,258 | 25 | 0 | 4,866 | 54 | 693 | 247 | 2,718 | 70 | 705 | 18.5 |

**docked**

| controller | served (city) | failed requests | mean walk m | redirects | bikes moved | swaps | van km | kWh | cost USD | kg CO2e | decisions | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DoNothing | 95.1% ± 0.8% | 2,272 | 2 | 24,452 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| StaticTSP 3 h | 96.7% ± 0.7% | 1,543 | 2 | 23,081 | 3,461 | 0 | 1,028 | 354 | 2,462 | 100 | 118 | 1.7 |
| StaticTSP 1 h | 96.8% ± 0.8% | 1,508 | 2 | 20,291 | 3,923 | 0 | 846 | 295 | 2,476 | 83 | 296 | 1.6 |
| GreedyDeficit | 97.0% ± 0.7% | 1,370 | 1 | 18,957 | 4,335 | 0 | 980 | 335 | 2,765 | 95 | 880 | 11.9 |
| GreedyDeficit + swaps | 97.0% ± 0.6% | 1,370 | 1 | 18,866 | 4,237 | 67 | 977 | 335 | 2,767 | 95 | 862 | 11.7 |

## Gains against doing nothing (paired by seed, ± s.d.)

| mode | controller | Δ served (pp) | extra trips served | cost per extra trip (USD) | kWh per shift | bikes moved per van km |
|---|---|---|---|---|---|---|
| dockless | StaticTSP 3 h | +2.35 ± 0.12 | 1,468 | 1.76 | 37 | 6.2 |
| dockless | StaticTSP 1 h | +1.95 ± 0.11 | 1,212 | 1.98 | 30 | 7.9 |
| dockless | GreedyDeficit | +2.20 ± 0.09 | 1,370 | 1.99 | 36 | 6.9 |
| dockless | GreedyDeficit + swaps | +2.31 ± 0.09 | 1,452 | 1.87 | 35 | 7.0 |
| docked | StaticTSP 3 h | +1.58 ± 0.07 | 729 | 3.38 | 51 | 3.4 |
| docked | StaticTSP 1 h | +1.66 ± 0.05 | 764 | 3.24 | 42 | 4.6 |
| docked | GreedyDeficit | +1.96 ± 0.08 | 902 | 3.07 | 48 | 4.4 |
| docked | GreedyDeficit + swaps | +1.95 ± 0.14 | 902 | 3.07 | 48 | 4.4 |

## Findings

1. **Every baseline beats doing nothing by about 2 points of service** with one van working 06:00–22:00:
   - dockless: 78.1 % → 80.4 % (StaticTSP 3 h, Greedy with swaps), 80.3 % (Greedy);
   - docked: 95.1 % → 96.7 % (StaticTSP 3 h), 97.0 % (Greedy).

   Paired by seed the gains are precise (± 0.1 points): about 1,200–1,450 extra trips served per
   week dockless, and 730–900 docked.
2. **Greedy is the best baseline in docked mode**: +1.96 points and 22 % fewer redirects than doing
   nothing. In dockless mode the 3-hourly StaticTSP (+2.35) and Greedy with swaps (+2.31) are on
   par. Replanning every hour does not help StaticTSP dockless (+1.95): plans are cut short before
   the far zones are served.
3. **Cost and energy.**
   - The van costs USD 2,400–2,800 a week: wages for working time, energy and maintenance. That is
     USD 1.8–2.0 per extra trip dockless and 3.1–3.4 docked.
   - It uses 30–51 kWh per shift, below its 68 kWh, and no run is flagged: the vans return to the
     depot every night, so the always-charged assumption (D20) holds.
   - An earlier version of the controllers never returned to the depot. Every run was then
     flagged, with 350–500 kWh between depot visits, and the van drove a third more for the same
     service (#161).
4. **Battery swaps barely matter here.** With 20 % e-bikes and a 60 km range, few e-bikes run low
   in a week (54–67 swaps), and low batteries are not why requests fail. Swaps will matter in
   longer runs and with more e-bikes.
5. **What limits dockless service is walking**, not the number of bikes per zone. Most failures are
   users declining the walk to the nearest bike (spec §6.5, #32).
   - An earlier StaticTSP that dropped every bike on one node per zone served *fewer* users than
     doing nothing. Spreading drops over the zone's busiest nodes was needed (#152).
   - Placement within a zone is where an RL agent can add the most.
6. **Speed.** A week takes 0.4 s without operations, 2–3 s with StaticTSP and 12–20 s with Greedy,
   which computes a forecast at each of its 700–900 decisions.
