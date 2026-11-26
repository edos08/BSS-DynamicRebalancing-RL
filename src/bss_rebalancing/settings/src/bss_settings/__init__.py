"""Personal settings (API keys, tracking accounts, ...) kept outside the repository.

A setting is resolved from, in order of priority:

1. the process environment (``export NAME=value`` or ``NAME=value command``);
2. ``.env`` in the current directory (a per-checkout file, git-ignored);
3. the user file: ``$BSS_ENV_FILE`` if set, else ``$XDG_CONFIG_HOME/bss-rebalancing/.env``
   (default ``~/.config/bss-rebalancing/.env``).

Files hold ``NAME=value`` lines. ``export``, quotes, blank lines and ``#`` comments are allowed,
so the same file can also be ``source``d by a shell script. Any name can be added; the ones the
project uses are listed in :data:`KNOWN` and documented in ``.env.example`` at the repo root.

Values are never logged: :func:`report` only says whether a setting is set and where from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["KNOWN", "Setting", "get", "require", "apply_to_environ", "env_files", "read_env_file",
           "resolve", "report"]


@dataclass(frozen=True)
class Setting:
    name: str
    description: str
    secret: bool = False
    used_by: str = ""


#: Settings the project reads. Others may be added freely to the files.
KNOWN: dict[str, Setting] = {s.name: s for s in (
    Setting("TOMTOM_API_KEY", "TomTom Routing API key (traffic collection, issue #17)", True,
            "bss-scenario traffic collect"),
    Setting("GOOGLE_MAPS_API_KEY", "Google Routes API key (travel-time cross-check, issue #58)", True,
            "traffic validation"),
    Setting("WANDB_API_KEY", "Weights & Biases API key (https://wandb.ai/authorize)", True,
            "bss-train / train_grid --wandb"),
    Setting("WANDB_ENTITY", "W&B user or team that owns the runs (default: your W&B default entity)",
            False, "bss-train / train_grid --wandb"),
    Setting("WANDB_PROJECT", "W&B project name (default: bss-rebalancing)", False,
            "bss-train / train_grid --wandb"),
    Setting("WANDB_MODE", "W&B mode: online (default) or offline", False, "wandb"),
)}


def _user_file() -> Path:
    if os.environ.get("BSS_ENV_FILE"):
        return Path(os.environ["BSS_ENV_FILE"]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "bss-rebalancing" / ".env"


def env_files() -> list[Path]:
    """Settings files, highest priority first (they may not exist)."""
    return [Path.cwd() / ".env", _user_file()]


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines; a missing file is empty."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        out[name] = value
    return out


def resolve(name: str) -> tuple[str | None, str | None]:
    """``(value, source)``; source is ``"environment"`` or a file path, ``(None, None)`` if unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value, "environment"
    for path in env_files():
        value = read_env_file(path).get(name, "").strip()
        if value:
            return value, str(path)
    return None, None


def get(name: str, default: str | None = None) -> str | None:
    """The value of a setting, or ``default`` if it is not set anywhere."""
    value, _ = resolve(name)
    return value if value is not None else default


def require(name: str) -> str:
    """The value of a setting; a clear error saying where to put it if it is missing."""
    value = get(name)
    if value is None:
        hint = KNOWN[name].description if name in KNOWN else name
        raise RuntimeError(
            f"{name} is not set ({hint}). Add `{name}=...` to {_user_file()} (or ./.env, or the "
            f"environment); see .env.example. Never commit it.")
    return value


def apply_to_environ(names: list[str] | None = None) -> list[str]:
    """Copy settings from the files into ``os.environ`` where not already set.

    For libraries that read the environment themselves (e.g. ``wandb`` reads ``WANDB_API_KEY``).
    Returns the names that were set. ``names=None`` applies every name in the files.
    """
    found: dict[str, str] = {}
    for path in reversed(env_files()):          # lowest priority first, overwritten by higher
        found.update(read_env_file(path))
    applied = []
    for name, value in found.items():
        if (names is None or name in names) and value and not os.environ.get(name):
            os.environ[name] = value
            applied.append(name)
    return applied


def report() -> list[dict]:
    """Known settings plus any other names in the files: set or not, and where from. No values."""
    names = list(KNOWN)
    for path in env_files():
        names += [n for n in read_env_file(path) if n not in names]
    rows = []
    for name in names:
        _, source = resolve(name)
        s = KNOWN.get(name)
        rows.append({"name": name, "set": source is not None, "source": source,
                     "description": s.description if s else "(custom)",
                     "secret": bool(s and s.secret)})
    return rows
