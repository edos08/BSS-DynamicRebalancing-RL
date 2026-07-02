"""
Preprocessing manifests — a paper trail so a run can be reproduced later
without playing archaeologist through shell history. Written once, at the
END of a successful pipeline run. A manifest for a failed run is worse than
no manifest at all, because it lies to you with confidence.
"""

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from preprocessing.config import PreprocessingConfig
from preprocessing.core.sources import Source


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"  # don't crash a manifest over a version lookup


def _file_sha256(path: str) -> str:
    """Hash of sources.json at run time. If this doesn't match the current
    file later, the manifest is describing a source that no longer exists —
    someone edited sources.json since. Better to know than to assume."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return "unreadable"


def build_manifest(
    config: PreprocessingConfig,
    source: Source,
    steps_run: List[str],
    started_at: datetime,
) -> dict:
    finished_at = datetime.now(timezone.utc)

    return {
        "manifest_version": 1,
        "generated_at_utc": finished_at.isoformat(),
        "run_duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "steps_run": steps_run,

        "source": {
            "id": source.id,
            "definition": asdict(source) if is_dataclass(source) else vars(source),
            "sources_json_path": os.path.abspath(config.sources_json),
            "sources_json_sha256": _file_sha256(config.sources_json),
        },

        "config": {
            "data_path": os.path.abspath(config.data_path),
            "cell_size": config.cell_size,
            "interpolation_radius": config.interpolation_radius,
            "user_radius": config.user_radius,
            "num_time_slots": config.num_time_slots,
            "days_of_week": config.days_of_week,
            "nodes_to_remove": config.nodes_to_remove,
        },

        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "polars": _pkg_version("polars"),
            "osmnx": _pkg_version("osmnx"),
            "networkx": _pkg_version("networkx"),
            "numpy": _pkg_version("numpy"),
        },

        "reproduce_cmd": (
            f"bss-preprocess --data-path {config.data_path} "
            f"--source {source.id} --sources-json {config.sources_json}"
        ),
    }


def write_manifest(config: PreprocessingConfig, source: Source, steps_run: List[str], started_at: datetime) -> str:
    manifest = build_manifest(config, source, steps_run, started_at)

    os.makedirs(config.utils_path, exist_ok=True)
    manifest_path = os.path.join(config.utils_path, "preprocessing_manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nManifest saved to {manifest_path}")
    return manifest_path