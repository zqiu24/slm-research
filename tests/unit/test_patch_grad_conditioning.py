# tests/unit/test_patch_grad_conditioning.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_GRAD_CONDITIONING", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.grad_conditioning", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.grad_conditioning", None)


def test_patch_registers_without_owning_setup_symbol():
    importlib.import_module("src.patches.grad_conditioning")
    reg = registered_patches()
    assert "grad_conditioning" in reg
    # runtime wrapper: declares no static target (composes with the other
    # setup_model_and_optimizer wrappers), so it never owns that symbol.
    assert all("setup_model_and_optimizer" not in t for t in reg["grad_conditioning"].targets)


def test_select_linear_grad_targets_picks_named_2d_weights():
    """Pure selection logic: name must match a wanted projection AND the module
    must expose a 2D .weight. CPU-testable without Megatron."""
    import torch

    from src.patches.grad_conditioning import select_linear_grad_targets

    class FakeLinear:
        def __init__(self, shape):
            self.weight = torch.zeros(*shape)

    mods = {
        "decoder.layers.0.self_attention.linear_q": FakeLinear((8, 8)),
        "decoder.layers.5.self_attention.linear_v": FakeLinear((8, 8)),
        "decoder.layers.9.mlp.linear_fc2": FakeLinear((8, 16)),
        "decoder.layers.1.mlp.router": FakeLinear((4, 8)),  # name does not match
        "decoder.layers.2.mlp.linear_fc1_ln": object(),  # matches fragment, no .weight
    }
    targets = select_linear_grad_targets(mods.items(), max_targets=8)
    labels = {t["label"] for t in targets}
    assert any("linear_q" in lbl for lbl in labels)
    assert any("linear_v" in lbl for lbl in labels)
    assert any("linear_fc2" in lbl for lbl in labels)
    assert not any("router" in lbl for lbl in labels)
    assert len(targets) == 3  # the no-match and the weight-less module are dropped


def test_log_grad_conditioning_logs_all_four_metrics(monkeypatch):
    """Reads main_grad and logs grad_cond/<label>/{condition_number, stable_rank,
    sigma_max_over_median, effective_rank} for a heavy-tailed weight gradient."""
    import sys as _sys
    import types

    import torch

    from src.patches.grad_conditioning import _log_grad_conditioning

    captured = {}
    fake_wandb = types.SimpleNamespace(run=object(), log=lambda d, step=None: captured.update(d))
    monkeypatch.setitem(_sys.modules, "wandb", fake_wandb)

    torch.manual_seed(0)
    # a ~rank-4 dominant 32x32 gradient (heavy-tailed spectrum)
    g = torch.randn(32, 4) @ torch.randn(4, 32) + 0.01 * torch.randn(32, 32)
    param = torch.nn.Parameter(torch.zeros(32, 32))
    param.main_grad = g
    target = {"label": "decoder.layers.0.mlp.linear_fc2", "param": param}

    _log_grad_conditioning([target], iteration=0)

    layer = "decoder.layers.0.mlp.linear_fc2"
    raw = f"grad_cond/{layer}"
    upd = f"grad_update/{layer}"
    for metric in ("condition_number", "stable_rank", "sigma_max_over_median", "effective_rank"):
        # raw (pre-orthogonalization) AND post-NS update spectrum both logged
        assert f"{raw}/{metric}" in captured
        assert f"{upd}/{metric}" in captured
    # ~rank-4 structure -> raw effective/stable rank well below the 32 ambient dims
    assert captured[f"{raw}/effective_rank"] < 10.0
    assert captured[f"{raw}/stable_rank"] < 10.0
    # Newton-Schulz whitens the update: condition number collapses toward ~1,
    # well below the heavy-tailed raw gradient's.
    assert captured[f"{upd}/condition_number"] < captured[f"{raw}/condition_number"] / 3.0
    assert captured[f"{upd}/effective_rank"] > captured[f"{raw}/effective_rank"]


def test_interval_falls_back_to_poet_interval_for_consistency():
    """The probe samples at the same cadence as the POET probe: defaults to 2000,
    follows SLM_POET_GRAD_CONDITIONING_INTERVAL when set, but an explicit
    SLM_GRAD_CONDITIONING_INTERVAL always wins."""
    from src.patches.grad_conditioning import _resolve_interval

    assert _resolve_interval({}) == 2000
    assert _resolve_interval({"SLM_POET_GRAD_CONDITIONING_INTERVAL": "500"}) == 500
    assert _resolve_interval({"SLM_GRAD_CONDITIONING_INTERVAL": "100"}) == 100
    # explicit generic var overrides the POET fallback
    assert (
        _resolve_interval(
            {
                "SLM_GRAD_CONDITIONING_INTERVAL": "100",
                "SLM_POET_GRAD_CONDITIONING_INTERVAL": "500",
            }
        )
        == 100
    )


def test_install_grad_conditioning_on_setup_wraps_and_hooks():
    """Wrapping setup_model_and_optimizer returns the built tuple unchanged AND
    installs a step-hook that still calls the original .step."""
    import torch

    from src.patches.grad_conditioning import _install_grad_conditioning_on_setup

    class FakeOpt:
        def __init__(self):
            self.n = 0

        def step(self, *a, **k):
            self.n += 1

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(8, 8)  # name matches -> 1 target

    model, opt = FakeModel(), FakeOpt()

    def orig_setup(*a, **k):
        return model, opt, "sched"

    wrapped = _install_grad_conditioning_on_setup(orig_setup, interval=1)
    m, o, s = wrapped()

    assert (m, o, s) == (model, opt, "sched")
    # no main_grad/.grad on the fresh Linear -> logs 'no grad', never crashes
    o.step()
    o.step()
    assert opt.n == 2  # underlying .step still runs through the wrapper
