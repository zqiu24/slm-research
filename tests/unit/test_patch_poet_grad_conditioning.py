# tests/unit/test_patch_poet_grad_conditioning.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_POET_GRAD_CONDITIONING", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)


def test_patch_registers_with_unique_target():
    importlib.import_module("src.patches.poet_grad_conditioning")
    reg = registered_patches()
    assert "poet_grad_conditioning" in reg
    # unique label, NOT the real get_megatron_optimizer symbol owned by poet_optimizer_setup
    assert all("get_megatron_optimizer" not in t for t in reg["poet_grad_conditioning"].targets)


def test_select_target_params_picks_representative_blocks():
    """Pure selection logic is CPU-testable without Megatron."""
    import torch

    from src.patches.poet_grad_conditioning import select_target_params

    class FakePOET:
        def __init__(self, name, b):
            self.block_size_in = b
            self.block_size_out = b
            # one block's worth of upper-tri entries: b*(b-1)/2
            self.oft_R_in = torch.zeros(1, b * (b - 1) // 2)
            self.oft_R_out = torch.zeros(1, b * (b - 1) // 2)
            self._name = name

    layers = {
        "decoder.layers.0.self_attention.linear_q": FakePOET("q0", 4),
        "decoder.layers.5.self_attention.linear_v": FakePOET("v5", 4),
        "decoder.layers.9.mlp.linear_fc2": FakePOET("down9", 4),
        "decoder.layers.1.mlp.linear_no_match": FakePOET("x1", 4),
    }
    targets = select_target_params(layers.items(), max_targets=8)
    labels = {t["label"] for t in targets}
    # q/v/down projections selected (both R_in and R_out factors), the no-match dropped
    assert any("linear_q" in lbl for lbl in labels)
    assert any("linear_v" in lbl for lbl in labels)
    assert any("linear_fc2" in lbl for lbl in labels)
    assert not any("linear_no_match" in lbl for lbl in labels)
    # each selected layer contributes its R_in and R_out factor
    assert all(t["factor"] in ("R_in", "R_out") for t in targets)


def test_install_conditioning_on_setup_wraps_and_hooks():
    """The probe wraps setup_model_and_optimizer (not get_megatron_optimizer, which
    poet_optimizer_setup bypasses on the POET path): it must return the built
    (model, optimizer, scheduler) unchanged AND install a step-hook on the
    optimizer that still calls the original .step."""
    import torch

    from src.patches.poet_grad_conditioning import _install_conditioning_on_setup

    class FakeOpt:
        def __init__(self):
            self.n = 0

        def step(self, *a, **k):
            self.n += 1

    class FakePOET(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.block_size_in = b
            self.block_size_out = b
            self.oft_R_in = torch.nn.Parameter(torch.zeros(1, b * (b - 1) // 2))
            self.oft_R_out = torch.nn.Parameter(torch.zeros(1, b * (b - 1) // 2))

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = FakePOET(4)  # name matches _WANTED -> 2 targets (R_in/R_out)

    model, opt = FakeModel(), FakeOpt()

    def orig_setup(*a, **k):
        return model, opt, "sched"

    wrapped = _install_conditioning_on_setup(orig_setup, interval=1)
    m, o, s = wrapped()

    # returns the built tuple unchanged
    assert (m, o, s) == (model, opt, "sched")
    # step-hook installed (no main_grad/.grad on the fakes -> logs 'no grad', no crash)
    o.step()
    o.step()
    assert opt.n == 2  # underlying .step still runs through the wrapper


def test_log_conditioning_also_logs_muon_update_spectrum(monkeypatch):
    """_log_conditioning must log poet_update/<label>/cond_orthogonalized — the
    Muon-update spectrum (~1) — alongside the heavy-tailed raw-grad
    poet_cond/<label>/condition_number. The contrast is the only way to SEE Muon's
    preconditioning (the raw-grad probe alone never can)."""
    import sys
    import types

    import torch

    from src.patches.poet_grad_conditioning import _log_conditioning

    captured = {}
    fake_wandb = types.SimpleNamespace(run=object(), log=lambda d, step=None: captured.update(d))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    b, ne = 16, 16 * 15 // 2
    torch.manual_seed(0)
    v = torch.randn(2, ne)
    v[:, :3] *= 6.0  # heavy-tailed raw gradient (the (n_blocks, n_elems) vec form)
    param = torch.nn.Parameter(torch.zeros(2, ne))
    param.main_grad = v
    target = {"label": "x", "factor": "R_in", "param": param, "block_size": b, "layer": None}

    _log_conditioning([target], iteration=0)

    assert "poet_cond/x/condition_number" in captured
    assert "poet_update/x/cond_orthogonalized" in captured
    assert captured["poet_cond/x/condition_number"] > 10.0  # raw grad heavy-tailed
    assert captured["poet_update/x/cond_orthogonalized"] < 5.0  # Muon update flattened
    # full post-orthogonalization spectral stats (not just the condition number),
    # plus effective_rank on the raw side, so both spectra carry the same metrics.
    for metric in ("stable_rank", "effective_rank", "sigma_max_over_median"):
        assert f"poet_cond/x/{metric}" in captured
        assert f"poet_update/x/{metric}" in captured
    assert "poet_cond/x/effective_rank" in captured
    # NS whitens the skew update: effective rank rises vs the heavy-tailed raw grad
    assert captured["poet_update/x/effective_rank"] > captured["poet_cond/x/effective_rank"]
