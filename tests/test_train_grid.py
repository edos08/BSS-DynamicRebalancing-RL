"""Smoke test of the DQN training loop on GridZoneEnv (issue #166), on CPU with a synthetic bundle."""
import json

import pytest

pytest.importorskip("bss_gym")

from bss_sim.synthetic import make_bundle  # noqa: E402
from rl_training import train_grid  # noqa: E402


def test_train_grid_runs_and_stores_smdp_transitions(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path / "b", n_side=16, n_stations=6, weekday_trips=600, seed=1)
    pushed = []
    from rl_training.memory import ReplayBuffer
    orig = ReplayBuffer.push

    def spy(self, state, action, reward, next_state, done, steps=None, terminal=None):
        pushed.append((reward, steps, done, terminal))
        return orig(self, state, action, reward, next_state, done, steps=steps, terminal=terminal)
    monkeypatch.setattr(ReplayBuffer, "push", spy)
    res = tmp_path / "results"
    assert train_grid.main(["--bundle", str(bundle), "--episodes", "2", "--days", "0.15", "--device", "cpu",
                            "--fleet", "60", "--train-dates", "2024-09-17", "--eval-dates", "2024-09-18",
                            "--eval-seeds", "1", "--results-path", str(res), "--run-id", "7"]) == 0
    run = res / "run_007"
    for ep in (0, 1):
        d = run / "training" / f"episode_{ep:03d}"
        assert {"scalars.json", "timeslot_metrics.csv", "step_data.pkl.gz", "cell_subgraph.gpickle"} <= \
            {p.name for p in d.iterdir()}
    assert (run / "models" / "model_final.pt").exists()
    assert (run / "validation" / "final_episode_001" / "episode_000" / "scalars.json").exists()
    cfg = json.loads((run / "training" / "config.json").read_text())
    assert cfg["hyperparameters"]["smdp"] is True and cfg["hyperparameters"]["gamma"] == 0.95
    assert all(s >= 1 for _, s, _, _ in pushed) and any(s > 1 for _, s, _, _ in pushed)   # multi-step actions
    assert not any(t for *_, t in pushed)              # time limit: truncated, never terminal
    assert pushed[-1][2] is True                        # the last transition of an episode is done

    wa = pytest.importorskip("results_webapp.data_loader")   # the webapp reads the run
    assert wa.get_available_episodes(run, "training") == [0, 1]
    assert wa.get_available_episodes(run, "training") == [0, 1]
    assert len(wa.load_summary_data(run, "training")) == 2


def test_resume_continues_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    bundle = make_bundle(tmp_path / "b", n_side=14, n_stations=5, weekday_trips=500, seed=2)
    common = ["--bundle", str(bundle), "--episodes", "3", "--days", "0.08", "--device", "cpu", "--fleet", "50",
              "--train-dates", "2024-09-17", "--eval-dates", "2024-09-18", "--eval-seeds", "1",
              "--val-every", "2", "--results-path", str(tmp_path / "res")]
    assert train_grid.main(common + ["--run-id", "1"]) == 0
    assert train_grid.main(common + ["--run-id", "2", "--stop-after", "2"]) == 0
    run2 = tmp_path / "res" / "run_002"
    assert not (run2 / "models" / "model_final.pt").exists() and (run2 / "checkpoints" / "last.pt").exists()
    assert train_grid.main(common + ["--run-id", "2", "--resume"]) == 0
    for ep in range(3):
        a = json.loads((tmp_path / "res" / "run_001" / "training" / f"episode_{ep:03d}" / "scalars.json").read_text())
        b = json.loads((run2 / "training" / f"episode_{ep:03d}" / "scalars.json").read_text())
        assert a == b, ep                                # the resumed run is identical to the straight one
    m1 = torch.load(tmp_path / "res" / "run_001" / "models" / "model_final.pt", weights_only=True)
    m2 = torch.load(run2 / "models" / "model_final.pt", weights_only=True)
    assert all(torch.equal(m1["train_model"][k], m2["train_model"][k]) for k in m1["train_model"])
    assert (run2 / "models" / "model_best.pt").exists()
    assert (run2 / "validation" / "episode_001" / "episode_000" / "scalars.json").exists()
    import pandas as pd
    assert pd.read_csv(run2 / "training" / "train_summary.csv")["episode"].tolist() == [0, 1, 2]
    with pytest.raises(SystemExit):
        train_grid.main(common + ["--run-id", "3", "--resume"])     # nothing to resume
