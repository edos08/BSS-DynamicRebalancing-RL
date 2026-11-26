"""RNG state restore after torch.load(map_location=...) (issue #180)."""
import pytest
import torch

from rl_training.checkpoint import rng_state, set_rng_state


def _moved(state, device):
    out = dict(state)
    out["torch"] = state["torch"].to(device)
    if "cuda" in state:
        out["cuda"] = [s.to(device) for s in state["cuda"]]
    return out


def test_restore_roundtrip_cpu():
    state = rng_state()
    a = torch.rand(3)
    set_rng_state(state)
    assert torch.equal(torch.rand(3), a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_restore_after_loading_on_cuda():
    """A checkpoint loaded with map_location='cuda' has its CPU RNG state on the GPU."""
    state = rng_state()
    a, b = torch.rand(3), torch.rand(3, device="cuda")
    set_rng_state(_moved(state, "cuda"))
    assert torch.equal(torch.rand(3), a) and torch.equal(torch.rand(3, device="cuda"), b)


def test_more_saved_cuda_states_than_devices(monkeypatch):
    state = rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)] * 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    seen = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda s, i: seen.append(i))
    set_rng_state(state)
    assert seen == [0]
