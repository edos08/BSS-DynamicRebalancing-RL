"""Compare the baseline controllers on a scenario (issue #154).

Usage:
    python scripts/run_baselines.py data/scenarios/cambridge-2024 --days 7 --seeds 3 \
        --out docs/sim-v2/reports

For each operation mode (dockless, docked) and seed, runs one episode per controller from
``--start`` (Poisson demand, default fleet, one electric van) and records the KPIs. Writes
``baselines.json`` (every episode) and prints a Markdown table of means and spreads over seeds.
Docked runs start from bikes spread by dock capacity (``initial_distribution: uniform``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bss_policies import DoNothing, GreedyDeficit, StaticTSP, run_episode
from bss_sim.config import SimConfig
from bss_sim.simulator import Simulator

CONTROLLERS = {
    "DoNothing": lambda: DoNothing(),
    "StaticTSP 3 h": lambda: StaticTSP(every_h=3),
    "StaticTSP 1 h": lambda: StaticTSP(every_h=1),
    "GreedyDeficit": lambda: GreedyDeficit(),
    "GreedyDeficit + swaps": lambda: GreedyDeficit(swap_below=0.25),
}
MODES = {
    "dockless": {"mode": "dockless"},
    "docked": {"mode": "docked", "fleet": {"initial_distribution": "uniform"}},
}
COLUMNS = (("city_service_level", "served (city)", "{:.1%}"), ("failed", "failed requests", "{:,.0f}"),
           ("walk_m_mean", "mean walk m", "{:.0f}"), ("redirects", "redirects", "{:,.0f}"),
           ("bikes_moved", "bikes moved", "{:,.0f}"), ("batteries_swapped", "swaps", "{:,.0f}"),
           ("truck_km", "van km", "{:,.0f}"), ("truck_kwh", "kWh", "{:,.0f}"),
           ("operating_cost_usd", "cost USD", "{:,.0f}"), ("kgco2e", "kg CO2e", "{:,.0f}"))


def run(bundle: Path, days: int, seeds: int, start: str, modes: list[str], controllers: list[str]) -> list[dict]:
    rows = []
    for mode in modes:
        sim = Simulator.from_scenario(bundle, SimConfig.from_dict(MODES[mode]))
        for seed in range(seeds):
            for name in controllers:
                res = run_episode(sim, CONTROLLERS[name](), seed=seed, start=start, duration_s=days * 86400.0)
                k = {key: v for key, v in res.kpis.items() if key != "trucks"}
                rows.append({"mode": mode, "controller": name, "seed": seed, "params": res.controller,
                             "decisions": res.decisions, "wall_s": res.wall_s, "kpis": k})
                print(f"{mode:8s} seed {seed} {name:24s} served {k['city_service_level']:.3f} "
                      f"moved {k['bikes_moved']:5d} km {k['truck_km']:7.1f} wall {res.wall_s:6.1f}s", flush=True)
    return rows


def table(rows: list[dict]) -> str:
    out = []
    for mode in dict.fromkeys(r["mode"] for r in rows):
        out += [f"**{mode}**", "", "| controller | " + " | ".join(c[1] for c in COLUMNS) + " | decisions | wall s |",
                "|---|" + "---|" * (len(COLUMNS) + 2)]
        for name in dict.fromkeys(r["controller"] for r in rows if r["mode"] == mode):
            rs = [r for r in rows if r["mode"] == mode and r["controller"] == name]
            cells = []
            for key, _, fmt in COLUMNS:
                v = np.array([r["kpis"][key] for r in rs], dtype=float)
                cell = fmt.format(v.mean())
                if len(v) > 1 and key == "city_service_level":
                    cell += f" ± {v.std(ddof=1):.1%}"
                cells.append(cell)
            dec = np.mean([r["decisions"] for r in rs])
            wall = np.mean([r["wall_s"] for r in rs])
            out.append(f"| {name} | " + " | ".join(cells) + f" | {dec:,.0f} | {wall:.1f} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start", default="2024-09-16T00:00")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=list(CONTROLLERS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = run(args.bundle, args.days, args.seeds, args.start, args.modes, args.controllers)
    md = table(rows)
    print(md)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        meta = {"bundle": str(args.bundle), "days": args.days, "seeds": args.seeds, "start": args.start}
        (args.out / "baselines.json").write_text(json.dumps({"meta": meta, "episodes": rows}, indent=1) + "\n")
        (args.out / "baselines-table.md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
