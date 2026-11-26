"""Weights & Biases tracking of training runs (issue #171).

:class:`Tracker` logs the run configuration, per-episode and validation metrics, and stores the
run's model as a W&B **artifact**. The training loops upload only the best model (by validation),
once at the end of training, alias ``best`` (issue #174); checkpoints stay local.

It is a no-op unless enabled, so the training code calls it unconditionally. ``wandb`` is an
optional dependency (``pip install "bss-rl-training[tracking]"``).

**Credentials**: W&B reads its API key from ``WANDB_API_KEY`` or from ``wandb login``
(``~/.netrc``). Keep the key in ``~/.config/bss-rl/secrets.sh`` (``export WANDB_API_KEY=...``) and
``source`` it before training; nothing here prints or stores it. ``mode="offline"`` logs locally
(``wandb sync`` uploads later).

A resumed run (``--resume``) continues the same W&B run: its id is saved in the checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path


class Tracker:
    def __init__(self, enabled: bool = False, project: str = "bss-rebalancing", entity: str | None = None,
                 name: str | None = None, config: dict | None = None, run_dir: str | Path | None = None,
                 run_id: str | None = None, mode: str = "online", tags: tuple[str, ...] = (),
                 group: str | None = None):
        self.run = None
        self.name = name or "run"
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit('--wandb needs the wandb package: pip install "bss-rl-training[tracking]"') from exc
        self._wandb = wandb
        os.environ.setdefault("WANDB_SILENT", "true")          # also silences the login line
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        # silent: no wandb banner or upload chatter; one line of our own instead (errors still raise)
        self.run = wandb.init(project=project, entity=entity, name=name, config=config or {},
                              dir=str(run_dir) if run_dir else None, id=run_id, resume="allow", mode=mode,
                              tags=list(tags), group=group, settings=wandb.Settings(silent=True))
        where = self.run.url if mode != "offline" and self.run.url else f"offline, {self.run.dir}"
        print(f"W&B: {self.run.name} ({where})", flush=True)

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @property
    def id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def summary(self, values: dict) -> None:
        if self.run is not None:
            for k, v in values.items():
                self.run.summary[k] = v

    def log_model(self, path: str | Path, aliases: list[str], metadata: dict | None = None,
                  kind: str = "model") -> None:
        """Upload ``path`` as a new version of the artifact ``<run name>-<kind>`` with ``aliases``."""
        if self.run is None:
            return
        art = self._wandb.Artifact(f"{self.name}-{kind}".replace("/", "-"), type=kind, metadata=metadata or {})
        art.add_file(str(path))
        self.run.log_artifact(art, aliases=list(aliases))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
