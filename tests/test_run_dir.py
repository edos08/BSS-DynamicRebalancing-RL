"""Ask before overwriting an existing training run (issue #178)."""
import pytest

from rl_training.results import prepare_run_dir


def _make_run(root, run_id=2):
    run = root / f"run_{run_id:03d}"
    for sub in ("training/episode_000", "training/episode_001", "models", "checkpoints", "validation"):
        (run / sub).mkdir(parents=True)
    (run / "checkpoints" / "last.pt").write_text("old")
    return run


def test_new_or_empty_run_is_untouched(tmp_path):
    assert prepare_run_dir(tmp_path, 1, interactive=False) == tmp_path / "run_001"
    (tmp_path / "run_001").mkdir()
    prepare_run_dir(tmp_path, 1, interactive=False)
    assert (tmp_path / "run_001").exists()


def test_yes_deletes_the_whole_run(tmp_path, capsys):
    run = _make_run(tmp_path)
    prompts = []
    prepare_run_dir(tmp_path, 2, ask=lambda q: prompts.append(q) or "y", interactive=True)
    assert not run.exists()
    assert "Overwrite run 2?" in prompts[0] and "2 training episodes" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["", "n", "no", "whatever"])
def test_anything_but_yes_keeps_the_run_and_stops(tmp_path, answer):
    run = _make_run(tmp_path)
    with pytest.raises(SystemExit, match="another --run-id"):
        prepare_run_dir(tmp_path, 2, ask=lambda q: answer, interactive=True)
    assert (run / "checkpoints" / "last.pt").read_text() == "old"


def test_eof_counts_as_no(tmp_path):
    run = _make_run(tmp_path)

    def eof(_):
        raise EOFError
    with pytest.raises(SystemExit):
        prepare_run_dir(tmp_path, 2, ask=eof, interactive=True)
    assert run.exists()


def test_no_terminal_stops_without_asking(tmp_path):
    run = _make_run(tmp_path)
    with pytest.raises(SystemExit, match="--overwrite"):
        prepare_run_dir(tmp_path, 2, ask=lambda q: pytest.fail("asked"), interactive=False)
    assert run.exists()


def test_overwrite_flag_deletes_without_asking(tmp_path):
    run = _make_run(tmp_path)
    prepare_run_dir(tmp_path, 2, overwrite=True, ask=lambda q: pytest.fail("asked"), interactive=False)
    assert not run.exists()


def test_resume_keeps_everything(tmp_path):
    run = _make_run(tmp_path)
    prepare_run_dir(tmp_path, 2, resume=True, overwrite=True, ask=lambda q: pytest.fail("asked"))
    assert (run / "checkpoints" / "last.pt").exists()
