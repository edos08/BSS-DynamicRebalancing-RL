"""``bss-settings``: see which personal settings are configured (never prints values)."""
from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from bss_settings import _user_file, env_files, report


def _check(_args) -> int:
    print("Settings files (highest priority first; the environment wins over both):")
    for path in env_files():
        if path.exists():
            mode = stat.S_IMODE(path.stat().st_mode)
            warn = "  <- readable by others: chmod 600" if mode & 0o077 else ""
            print(f"  {path}  (found){warn}")
        else:
            print(f"  {path}  (missing)")
    print()
    rows = report()
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        where = r["source"] if r["set"] else "not set"
        print(f"  {'✓' if r['set'] else '·'} {r['name']:<{width}}  {where}\n      {r['description']}")
    return 0


def _init(args) -> int:
    path = Path(args.path).expanduser() if args.path else _user_file()
    if path.exists():
        print(f"{path} already exists; edit it directly.")
        return 0
    example = Path(args.example)
    if not example.exists():
        print(f"template {example} not found (run from the repo root or pass --example)", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(example.read_text())
    print(f"created {path} (mode 600) from {example}; fill in your values.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bss-settings", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="list settings, whether each is set and where from").set_defaults(func=_check)
    p = sub.add_parser("init", help="create your user settings file from .env.example")
    p.add_argument("--path", help="file to create (default: the user settings file)")
    p.add_argument("--example", default=".env.example")
    p.set_defaults(func=_init)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
