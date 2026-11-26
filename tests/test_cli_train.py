"""End-to-end: bss-train --one-validation (validation subprocess + best-model promotion) and bss-benchmark."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_train_one_validation(data_path, tmp_path):
    script = (
        "import sys\n"
        "from rl_training import train\n"
        "if __name__ == '__main__':\n"
        "    train.params['total_timeslots'] = 2\n"
        "    sys.argv = ['bss-train', '--data-path', %r, '--results-path', %r, '--num-episodes', '2',\n"
        "                '--max-num-bikes', '40', '--seed', '3', '--one-validation']\n"
        "    train.main()\n" % (data_path, str(tmp_path))
    )
    f = tmp_path / "run_train.py"
    f.write_text(script)
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    run = Path(tmp_path) / "run_000"
    # no validation happened during training (epsilon threshold ignored), only the final model was validated
    assert sorted(p.name for p in (run / "models" / "episodes").iterdir()) == ["episode_001"]
    assert (run / "models" / "best" / "trained_agent.pt").exists()
    scalars = json.loads((run / "validation" / "episode_001" / "episode_000" / "scalars.json").read_text())
    assert "mean_daily_failures" in scalars


def test_train_checkpoint_and_resume(data_path, tmp_path):
    def run(extra):
        script = (
            "import sys\n"
            "from rl_training import train\n"
            "if __name__ == '__main__':\n"
            "    train.params['total_timeslots'] = 2\n"
            "    sys.argv = ['bss-train', '--data-path', %r, '--results-path', %r, '--max-num-bikes', '40',\n"
            "                '--seed', '3', '--run-id', '5'] + %r\n"
            "    train.main()\n" % (data_path, str(tmp_path), extra)
        )
        f = tmp_path / "run_resume.py"
        f.write_text(script)
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return r.stdout

    run(["--num-episodes", "1"])
    ckpts = Path(tmp_path) / "run_005" / "checkpoints"
    assert (ckpts / "last.pt").exists() and (ckpts / "ckpt_episode_000.buffer.pt").exists()
    out = run(["--num-episodes", "2", "--resume"])
    assert "Resumed from ckpt_episode_000.pt: next episode 1" in out
    train = Path(tmp_path) / "run_005" / "training"
    assert (train / "episode_000" / "scalars.json").exists() and (train / "episode_001" / "scalars.json").exists()
    import pandas as pd
    assert pd.read_csv(train / "train_summary.csv")["episode"].tolist() == [0, 1]


def test_benchmark_cli(data_path, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "benchmark.run", "--data-path", data_path, "--results-path", str(tmp_path),
         "--max-num-bikes", "40", "--start-rebal-hour", "6"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cfg = json.loads((tmp_path / "run_000" / "benchmark" / "config.json").read_text())
    assert cfg["hyperparameters"]["starting_rebalancing_event"] == 6


def test_benchmark_all_exports():
    import benchmark
    for name in benchmark.__all__:
        assert hasattr(benchmark, name), name
