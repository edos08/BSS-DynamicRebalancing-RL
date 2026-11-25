"""Command line interface: ``bss-scenario <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bss_scenario import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bss-scenario",
        description="Build scenario bundles for the BSS simulator v2.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    trips = sub.add_parser(
        "trips",
        help="clean raw BlueBikes trip CSVs into monthly Parquet tables",
        description="Read raw monthly trip CSVs (2024 schema), flag problems, mark trips touching "
        "the city boundary and write <out>/YYYY-MM.parquet plus cleaning_report.json.",
    )
    trips.add_argument("--raw", nargs="+", required=True, type=Path,
                       help="raw CSV files, or directories containing *-tripdata.csv files")
    trips.add_argument("--boundary", required=True, type=Path, help="city boundary GeoJSON (WGS84)")
    trips.add_argument("--out", required=True, type=Path, help="output directory (e.g. <bundle>/trips)")
    trips.add_argument("--min-duration", type=float, default=60.0, help="seconds; shorter trips are flagged")
    trips.add_argument("--max-duration", type=float, default=10800.0, help="seconds; longer trips are flagged")
    trips.add_argument("--keep", choices=["touching_city", "all"], default="touching_city",
                       help="keep only trips with at least one end in the city (default) or all")
    trips.set_defaults(func=_cmd_trips)

    traffic = sub.add_parser("traffic", help="truck travel-time data (OD sample, routing queries, fit)")
    traffic_sub = traffic.add_subparsers(dest="traffic_command", metavar="<step>", required=True)
    sample = traffic_sub.add_parser("sample", help="draw the stratified OD sample for routing queries")
    sample.add_argument("--graph", required=True, type=Path, help="drive network GraphML (OSMnx)")
    sample.add_argument("--out", required=True, type=Path, help="output Parquet, e.g. data/raw/traffic/od_sample.parquet")
    sample.add_argument("--n-pairs", type=int, default=200)
    sample.add_argument("--seed", type=int, default=20241)
    sample.set_defaults(func=_cmd_traffic_sample)

    collect = traffic_sub.add_parser(
        "collect", help="query the TomTom Routing API for every OD pair x day type x hour (resumable)",
        description="Requires TOMTOM_API_KEY. Responses are cached; reruns only query what is missing.")
    collect.add_argument("--od", required=True, type=Path, help="OD sample Parquet (traffic sample)")
    collect.add_argument("--cache", required=True, type=Path, help="cache directory, e.g. data/raw/traffic/tomtom")
    collect.add_argument("--max-requests", type=int, default=2400, help="per run / per day (free tier: 2,500/day)")
    collect.add_argument("--max-qps", type=float, default=2.0)
    collect.add_argument("--daily", action="store_true",
                         help="keep running: one batch per day until everything is cached")
    collect.set_defaults(func=_cmd_traffic_collect)

    st = traffic_sub.add_parser("status", help="show how much of the collection is cached")
    st.add_argument("--od", required=True, type=Path)
    st.add_argument("--cache", required=True, type=Path)
    st.set_defaults(func=_cmd_traffic_status)
    return parser


def _cmd_traffic_collect(args: argparse.Namespace) -> int:
    import datetime as dt
    import json
    import time

    import polars as pl

    from bss_scenario.traffic import tomtom

    od = pl.read_parquet(args.od)
    fetch = tomtom.urllib_fetch(tomtom.api_key_from_env())
    while True:
        started = time.time()
        result = tomtom.collect(od, args.cache, fetch, max_requests=args.max_requests, max_qps=args.max_qps)
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        print(f"[{stamp}] {json.dumps(result.as_dict())}", flush=True)
        if result.stopped == "done" and result.remaining == 0:
            return 0
        if result.stopped == "too_many_errors":
            return 1
        if not args.daily:
            return 0
        time.sleep(max(0.0, 24 * 3600 + 300 - (time.time() - started)))


def _cmd_traffic_status(args: argparse.Namespace) -> int:
    import json

    import polars as pl

    from bss_scenario.traffic import tomtom

    print(json.dumps(tomtom.status(pl.read_parquet(args.od), args.cache), indent=2))
    return 0


def _cmd_traffic_sample(args: argparse.Namespace) -> int:
    from bss_scenario.traffic import ODSampleConfig, build_od_sample
    from bss_scenario.traffic.sample import summarise

    config = ODSampleConfig(n_pairs=args.n_pairs, seed=args.seed)
    sample = build_od_sample(args.graph, args.out, config)
    print(summarise(sample, config.bands_m))
    print(f"{sample.height} pairs written to {args.out}")
    return 0


def _expand(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*-tripdata.csv")) if p.is_dir() else [p])
    return files


def _cmd_trips(args: argparse.Namespace) -> int:
    from bss_scenario.trips import CleaningConfig, clean_trips

    missing = [str(p) for p in args.raw if not p.exists()]
    if missing:
        print(f"input not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    files = _expand(args.raw)
    if not files:
        print("no input CSV files found", file=sys.stderr)
        return 2
    config = CleaningConfig(args.min_duration, args.max_duration, args.keep)
    report = clean_trips(files, args.boundary, args.out, config)
    kept = sum(m["trips"] for m in report["kept_by_month"])
    valid = sum(m["valid"] for m in report["kept_by_month"])
    print(f"{report['read']['rows_out']:,} trips read, {kept:,} kept, {valid:,} valid; "
          f"{len(report['outputs'])} files written to {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
