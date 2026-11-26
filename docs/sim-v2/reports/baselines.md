# Baseline controllers (M5)

Reference controllers for the RL agents (spec §14, epic #42), on Cambridge 2024.

## Setup

- **Simulation**: one week from Mon 2024-09-16, 3 seeds. Demand from the fitted model (Poisson), 800
  bikes, one Ford E-Transit van starting at the depot with 15 bikes and 60 batteries. Travel at free
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
| DoNothing | 78.1% ± 0.7% | 12,711 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| StaticTSP 3 h | 80.3% ± 0.6% | 11,325 | 25 | 0 | 6,766 | 0 | 952 | 344 | 3,685 | 97 | 109 | 2.0 |
| StaticTSP 1 h | 79.9% ± 0.8% | 11,574 | 26 | 0 | 6,627 | 0 | 735 | 274 | 3,407 | 77 | 325 | 2.1 |
| GreedyDeficit | 80.2% ± 0.6% | 11,371 | 25 | 0 | 7,513 | 0 | 967 | 349 | 3,992 | 99 | 1,009 | 25.1 |
| GreedyDeficit + swaps | 80.2% ± 0.6% | 11,394 | 25 | 0 | 7,425 | 58 | 965 | 349 | 3,992 | 99 | 987 | 24.7 |

**docked**

| controller | served (city) | failed requests | mean walk m | redirects | bikes moved | swaps | van km | kWh | cost USD | kg CO2e | decisions | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DoNothing | 95.1% ± 0.8% | 2,272 | 2 | 24,452 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| StaticTSP 3 h | 96.7% ± 0.7% | 1,527 | 2 | 24,082 | 5,044 | 0 | 1,488 | 511 | 3,555 | 144 | 109 | 2.0 |
| StaticTSP 1 h | 96.8% ± 0.7% | 1,506 | 1 | 21,541 | 5,699 | 0 | 1,190 | 415 | 3,520 | 117 | 325 | 2.0 |
| GreedyDeficit | 97.1% ± 0.7% | 1,326 | 1 | 20,066 | 6,445 | 0 | 1,450 | 498 | 4,077 | 141 | 1,322 | 15.8 |
| GreedyDeficit + swaps | 97.1% ± 0.7% | 1,350 | 1 | 19,642 | 6,339 | 58 | 1,463 | 502 | 4,079 | 142 | 1,298 | 15.5 |

## Gains against doing nothing (paired by seed, ± s.d.)

| mode | controller | Δ served (pp) | extra trips served | cost per extra trip (USD) | bikes moved per van km |
|---|---|---|---|---|---|
| dockless | StaticTSP 3 h | +2.23 ± 0.13 | 1,386 | 2.66 | 7.1 |
| dockless | StaticTSP 1 h | +1.83 ± 0.18 | 1,136 | 3.00 | 9.0 |
| dockless | GreedyDeficit | +2.16 ± 0.11 | 1,340 | 2.98 | 7.8 |
| dockless | GreedyDeficit + swaps | +2.12 ± 0.10 | 1,317 | 3.03 | 7.7 |
| docked | StaticTSP 3 h | +1.61 ± 0.11 | 745 | 4.77 | 3.4 |
| docked | StaticTSP 1 h | +1.66 ± 0.05 | 766 | 4.59 | 4.8 |
| docked | GreedyDeficit | +2.05 ± 0.16 | 946 | 4.31 | 4.5 |
| docked | GreedyDeficit + swaps | +2.00 ± 0.11 | 922 | 4.42 | 4.3 |

## Findings

1. **Every baseline beats doing nothing, by about 2 points of service**, with one van:
   - dockless: 78.1 % → 80.3 % (StaticTSP 3 h), 80.2 % (Greedy);
   - docked: 95.1 % → 96.7 % (StaticTSP), 97.1 % (Greedy).

   Paired by seed the gains are precise (± 0.1–0.2 points): about 1,300–1,400 extra trips served
   per week dockless and 750–950 docked.
2. **Greedy is the best baseline in docked mode** (+2.05 points, 20 % fewer redirects than doing
   nothing). In dockless mode it ties with the 3-hourly StaticTSP. Replanning every hour does not
   help StaticTSP dockless (+1.8): plans are cut short before the far zones are served.
3. **Cost.** The van costs USD 3,400–4,100 a week (wages for working time, energy and maintenance).
   That is USD 2.7–3.0 per extra trip dockless and 4.3–4.8 docked. Energy is 270–510 kWh a week:
   a van never comes near its battery between depot visits (no energy flags).
4. **Battery swaps barely matter here.** With 20 % e-bikes and a 60 km range, few e-bikes run low
   in a week (58 swaps), and low batteries are not why requests fail. Swaps will matter in longer
   runs and with more e-bikes.
5. **What limits dockless service is walking**, not the number of bikes per zone. Most failures are
   users declining the walk to the nearest bike, about 22 % of city requests (spec §6.5, #32). An
   earlier StaticTSP that dropped every bike on one node per zone served *fewer* users than doing
   nothing. Spreading drops over the zone's busiest nodes was needed (#152). Placement within a
   zone is where an RL agent can add the most.
6. **Speed.** A week takes 0.4 s without operations, 2 s with StaticTSP and 16–25 s with Greedy.
   Greedy computes a forecast at each of its ~1,000–1,300 decisions.
