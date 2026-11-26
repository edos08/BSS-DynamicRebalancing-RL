"""Weights & Biases tracking (issue #171): no-op when disabled, offline run with artifacts when enabled."""
import pytest

from rl_training.tracking import Tracker


def test_disabled_tracker_is_a_no_op(tmp_path):
    t = Tracker(False)
    assert not t.enabled and t.id is None
    t.log({"a": 1}, step=0)
    t.summary({"b": 2})
    t.log_model(tmp_path / "missing.pt", ["best"])     # never touches the file
    t.finish()


def test_offline_run_logs_metrics_and_models(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    pytest.importorskip("bss_gym")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from bss_sim.synthetic import make_bundle
    from rl_training import train_grid, tracking

    logged, models, summaries = [], [], {}
    orig_log, orig_model, orig_summary = tracking.Tracker.log, tracking.Tracker.log_model, tracking.Tracker.summary

    def log(self, metrics, step=None):
        logged.append((step, dict(metrics)))
        return orig_log(self, metrics, step)

    def log_model(self, path, aliases, metadata=None, kind="model"):
        models.append((kind, tuple(aliases)))
        return orig_model(self, path, aliases, metadata, kind)

    def summary(self, values):
        summaries.update(values)
        return orig_summary(self, values)
    monkeypatch.setattr(tracking.Tracker, "log", log)
    monkeypatch.setattr(tracking.Tracker, "log_model", log_model)
    monkeypatch.setattr(tracking.Tracker, "summary", summary)

    bundle = make_bundle(tmp_path / "b", n_side=14, n_stations=5, weekday_trips=500, seed=3)
    common = ["--bundle", str(bundle), "--episodes", "2", "--days", "0.08", "--device", "cpu", "--fleet", "50",
              "--train-dates", "2024-09-17", "--eval-dates", "2024-09-18", "--eval-seeds", "1", "--val-every", "1",
              "--results-path", str(tmp_path / "res"), "--run-id", "4", "--wandb", "--wandb-mode", "offline"]
    assert train_grid.main(common + ["--stop-after", "1"]) == 0
    run_dir = tmp_path / "res" / "run_004"
    import torch
    wid = torch.load(run_dir / "checkpoints" / "ckpt_episode_000.pt", weights_only=False)["extra"]["wandb_id"]
    assert wid
    assert train_grid.main(common + ["--resume"]) == 0
    runs = sorted((run_dir / "wandb").glob("offline-run-*"))
    assert len(runs) == 2 and all(r.name.endswith(wid) for r in runs)   # the resume continued the same W&B run
    assert all(any(r.glob("*.wandb")) for r in runs)                      # real offline data written
    steps = {step for step, m in logged if "train/return" in m}
    assert steps == {0, 1} and any("val/city_service_level" in m for _, m in logged)
    assert ("checkpoint", ("latest", "episode-0")) in models and ("model", ("final",)) in models
    assert any(kind == "model" and "best" in aliases for kind, aliases in models)
    assert "eval/city_service_level" in summaries and "best/val_city_service_level" in summaries
