"""Personal settings outside the repo (issue #176)."""
import os
import re
import stat
from pathlib import Path

import pytest

import bss_settings
from bss_settings.cli import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated cwd and user file; no known setting in the environment."""
    cwd = tmp_path / "checkout"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("BSS_ENV_FILE", raising=False)
    for name in list(bss_settings.KNOWN) + ["MY_CUSTOM"]:
        monkeypatch.delenv(name, raising=False)
    user = tmp_path / "config" / "bss-rebalancing" / ".env"
    user.parent.mkdir(parents=True)
    return cwd, user


def test_parse_formats(tmp_path):
    f = tmp_path / "s.env"
    f.write_text("# comment\n\nexport A=1\nB = two \nC='quoted # not a comment'\nD=\"x\"\n"
                 "E=val # trailing\nnot a line\n1BAD=x\nEMPTY=\n")
    assert bss_settings.read_env_file(f) == {"A": "1", "B": "two", "C": "quoted # not a comment",
                                             "D": "x", "E": "val", "EMPTY": ""}
    assert bss_settings.read_env_file(tmp_path / "missing") == {}


def test_priority_environment_then_local_then_user(home, monkeypatch):
    cwd, user = home
    user.write_text("WANDB_ENTITY=from-user\nWANDB_PROJECT=user-project\n")
    assert bss_settings.resolve("WANDB_ENTITY") == ("from-user", str(user))
    (cwd / ".env").write_text("WANDB_ENTITY=from-local\n")
    assert bss_settings.get("WANDB_ENTITY") == "from-local"
    assert bss_settings.get("WANDB_PROJECT") == "user-project"     # falls through to the user file
    monkeypatch.setenv("WANDB_ENTITY", "from-env")
    assert bss_settings.resolve("WANDB_ENTITY") == ("from-env", "environment")
    assert bss_settings.get("WANDB_MODE", "online") == "online"


def test_empty_values_are_unset_and_bss_env_file(home, monkeypatch, tmp_path):
    _, user = home
    user.write_text("TOMTOM_API_KEY=\n")
    assert bss_settings.get("TOMTOM_API_KEY") is None
    other = tmp_path / "elsewhere.sh"
    other.write_text("export TOMTOM_API_KEY=abc\n")
    monkeypatch.setenv("BSS_ENV_FILE", str(other))
    assert bss_settings.require("TOMTOM_API_KEY") == "abc"


def test_require_error_says_where_without_values(home):
    with pytest.raises(RuntimeError, match=r"TOMTOM_API_KEY is not set .*\.env\.example"):
        bss_settings.require("TOMTOM_API_KEY")


def test_apply_to_environ_does_not_override(home, monkeypatch):
    _, user = home
    user.write_text("WANDB_API_KEY=k-file\nMY_CUSTOM=1\nWANDB_ENTITY=e\n")
    monkeypatch.setenv("WANDB_ENTITY", "already")
    assert bss_settings.apply_to_environ(["WANDB_API_KEY", "WANDB_ENTITY"]) == ["WANDB_API_KEY"]
    assert os.environ["WANDB_API_KEY"] == "k-file" and os.environ["WANDB_ENTITY"] == "already"
    assert "MY_CUSTOM" not in os.environ


def test_check_never_prints_values(home, capsys):
    _, user = home
    user.write_text("TOMTOM_API_KEY=supersecretvalue\nMY_CUSTOM=othersecret\n")
    user.chmod(0o644)
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "supersecretvalue" not in out and "othersecret" not in out
    assert "TOMTOM_API_KEY" in out and "MY_CUSTOM" in out and "chmod 600" in out


def test_init_creates_private_user_file(home, capsys):
    _, user = home
    assert main(["init", "--example", str(REPO / ".env.example")]) == 0
    assert user.exists() and stat.S_IMODE(user.stat().st_mode) == 0o600
    assert main(["init", "--example", str(REPO / ".env.example")]) == 0        # never overwrites
    assert bss_settings.get("TOMTOM_API_KEY") is None                         # template is empty


def test_example_lists_every_known_setting_with_no_values():
    example = bss_settings.read_env_file(REPO / ".env.example")
    assert set(example) == set(bss_settings.KNOWN)
    assert all(v == "" for v in example.values())


def test_example_and_env_files_are_git_ignored_correctly():
    lines = (REPO / ".gitignore").read_text().splitlines()
    assert ".env" in lines and "!.env.example" in lines


def test_tomtom_key_comes_from_settings(home):
    from bss_scenario.traffic import tomtom
    _, user = home
    user.write_text("TOMTOM_API_KEY=tt\n")
    assert tomtom.api_key_from_env() == "tt"


def test_no_personal_settings_in_tracked_text():
    """Guard: no real API keys or W&B entities committed next to the settings names."""
    import subprocess
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    pat = re.compile(r"\b(TOMTOM_API_KEY|GOOGLE_MAPS_API_KEY|WANDB_API_KEY|WANDB_ENTITY)\s*=\s*['\"]?[A-Za-z0-9_\-]{6,}")
    hits = []
    for f in files:
        if f == "tests/test_settings.py":       # this file's fake values
            continue
        p = REPO / f
        if p.suffix in {".py", ".md", ".sh", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".ini", ""} and p.is_file():
            try:
                if pat.search(p.read_text(errors="ignore")):
                    hits.append(f)
            except OSError:
                pass
    assert hits == []
