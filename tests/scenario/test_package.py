import ast
import pathlib

import pytest

import bss_scenario
from bss_scenario import cli

PKG_DIR = pathlib.Path(bss_scenario.__file__).parent
FORBIDDEN = {"gymnasium_env", "preprocessing", "rl_training", "benchmark", "results_webapp", "bss_sim", "torch"}


def test_version():
    assert bss_scenario.__version__.count(".") == 2


def test_cli_help_and_version(capsys):
    assert cli.main([]) == 0
    assert "bss-scenario" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert bss_scenario.__version__ in capsys.readouterr().out


def test_no_forbidden_imports():
    """Spec §3: bss_scenario must not depend on the simulator or the legacy packages."""
    for path in PKG_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in FORBIDDEN, f"{path.name} imports {name}"
