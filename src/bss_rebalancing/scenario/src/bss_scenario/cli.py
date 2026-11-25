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
    return parser


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
