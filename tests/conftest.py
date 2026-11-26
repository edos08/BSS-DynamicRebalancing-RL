import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth_data import build  # noqa: E402


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    build(str(root))
    return str(root)
