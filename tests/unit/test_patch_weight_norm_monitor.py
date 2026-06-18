# tests/unit/test_patch_weight_norm_monitor.py
import sys

import pytest

from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)


def test_parse_layer_selection_keywords_and_indices():
    from src.patches.weight_norm_monitor import parse_layer_selection

    assert parse_layer_selection("first,mid,last", 12) == {0, 6, 11}
    assert parse_layer_selection("0,5,11", 12) == {0, 5, 11}
    assert parse_layer_selection("-1", 12) == {11}  # negative wraps
    assert parse_layer_selection("99", 12) == set()  # out of range dropped
    assert parse_layer_selection(" first , 3 ", 12) == {0, 3}  # whitespace tolerant


def test_classify_linear_matches_fused_and_unfused_types_and_skips_others():
    from src.patches.weight_norm_monitor import classify_linear

    # fused names
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv") == (5, "qkv")
    assert classify_linear("decoder.layers.0.self_attention.linear_proj") == (0, "proj")
    assert classify_linear("module.decoder.layers.3.mlp.linear_fc1") == (3, "fc1")
    assert classify_linear("decoder.layers.7.mlp.linear_fc2") == (7, "fc2")
    # unfused names (--unfuse-qkv / --unfuse-fc1, e.g. head-aligned POET configs)
    assert classify_linear("decoder.layers.2.self_attention.linear_q") == (2, "q")
    assert classify_linear("decoder.layers.2.self_attention.linear_k") == (2, "k")
    assert classify_linear("decoder.layers.2.self_attention.linear_v") == (2, "v")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_gate") == (4, "fc1_gate")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_up") == (4, "fc1_up")
    # POET base-weight child module is NOT matched (name doesn't end in a type suffix)
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv.poet_linear") is None
    # embeddings / lm_head / norms not matched
    assert classify_linear("embedding.word_embeddings") is None
    assert classify_linear("output_layer") is None


def test_should_log_cadence_for_adam_and_poet():
    from src.patches.weight_norm_monitor import should_log

    # non-POET: every `interval`
    assert should_log(100, 100, poet=False, merge_period=0) is True
    assert should_log(150, 100, poet=False, merge_period=0) is False
    assert should_log(0, 100, poet=False, merge_period=0) is False
    # POET merge_period=1: base == W_eff every step -> gated only by interval
    assert should_log(50, 50, poet=True, merge_period=1) is True
    # POET merge_period=400: only on merge boundaries
    assert should_log(400, 100, poet=True, merge_period=400) is True
    assert should_log(100, 100, poet=True, merge_period=400) is False
    # POET merge_period=0: frozen base -> never
    assert should_log(100, 100, poet=True, merge_period=0) is False


def test_compute_matrix_norm_stats_values_and_rms():
    import math

    import torch

    from src.patches.weight_norm_monitor import compute_matrix_norm_stats

    # 2x3 matrix: rows have norms; cols have norms.
    w = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    stats = compute_matrix_norm_stats(w)

    # only the mean is emitted now (min/max/std commented out to cut W&B volume)
    assert set(stats["row"]) == {"mean"}
    # row norms = [3, 4]; row_rms = norm / sqrt(in_dim=3)
    assert stats["row"]["mean"] == pytest.approx(3.5)
    assert stats["row_rms"]["mean"] == pytest.approx(3.5 / math.sqrt(3))
    # col norms = [3, 4, 0]; col_rms = norm / sqrt(out_dim=2)
    assert stats["col"]["mean"] == pytest.approx(7.0 / 3.0)
    assert stats["col_rms"]["mean"] == pytest.approx((7.0 / 3.0) / math.sqrt(2))
    # raw RMS vectors are returned for histogram pooling
    assert stats["_row_rms_vec"].shape == (2,)
    assert stats["_col_rms_vec"].shape == (3,)


class _FakeMod:
    def __init__(self, weight):
        self.weight = weight


class _FakeChunk:
    """Minimal stand-in for a model chunk exposing named_modules()."""

    def __init__(self, named):
        self._named = named  # list[(name, module)]

    def named_modules(self):
        return iter(self._named)


def _fake_model():
    import torch

    named = [
        ("decoder.layers.0.self_attention.linear_qkv", _FakeMod(torch.ones(6, 4))),
        ("decoder.layers.0.mlp.linear_fc1", _FakeMod(torch.ones(8, 4) * 2)),
        ("decoder.layers.0.self_attention.linear_qkv.poet_linear", _FakeMod(torch.zeros(6, 4))),
        ("decoder.layers.1.self_attention.linear_qkv", _FakeMod(torch.ones(6, 4) * 3)),
        ("embedding.word_embeddings", _FakeMod(torch.ones(100, 4))),
    ]
    return [_FakeChunk(named)]


def test_collect_target_weights_filters_to_selected_layers_and_types():
    from src.patches.weight_norm_monitor import collect_target_weights

    got = collect_target_weights(_fake_model(), {0})
    labels = {(idx, mtype) for idx, mtype, _w in got}
    # layer 0 qkv + fc1 only; the poet_linear child, layer 1, and embeddings dropped
    assert labels == {(0, "qkv"), (0, "fc1")}


def test_log_weight_norms_emits_scalars_and_per_layer_histograms(monkeypatch):
    import sys
    import types

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update(d),
        Histogram=lambda x: ("HIST", len(x)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    opts = types.SimpleNamespace(num_layers=2, weight_norm_layers="0")
    from src.patches.weight_norm_monitor import _log_weight_norms

    _log_weight_norms(_fake_model(), iteration=100, opts=opts)

    # scalar keys for both matrices in layer 0, all four kinds (mean only now)
    assert "weightnorm/L0/qkv/row/mean" in captured
    assert "weightnorm/L0/qkv/col_rms/mean" in captured
    assert "weightnorm/L0/fc1/row/mean" in captured
    # min/max/std are no longer logged
    assert not any(k.endswith(("/min", "/max", "/std")) for k in captured)
    # per-layer pooled RMS histograms (one row + one col), tagged HIST
    assert captured["weightnorm/L0/row_rms_hist"][0] == "HIST"
    assert captured["weightnorm/L0/col_rms_hist"][0] == "HIST"
    # pooled row histogram length = qkv rows (6) + fc1 rows (8) = 14
    assert captured["weightnorm/L0/row_rms_hist"][1] == 14
    # layer 1 was not selected -> no keys for it
    assert not any(k.startswith("weightnorm/L1/") for k in captured)


def test_log_weight_norms_noop_when_wandb_run_is_none(monkeypatch):
    import sys
    import types

    captured = {}
    fake_wandb = types.SimpleNamespace(run=None, log=lambda d, step=None: captured.update(d))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    opts = types.SimpleNamespace(num_layers=2, weight_norm_layers="0")
    from src.patches.weight_norm_monitor import _log_weight_norms

    # must not raise even though no run is active
    _log_weight_norms(_fake_model(), iteration=100, opts=opts)
    # and must not have logged anything (the wandb.run is None guard short-circuits)
    assert not captured, f"unexpected log calls: {list(captured)[:3]}"


def test_wrapper_logs_after_inner_train_step_with_post_step_weights(monkeypatch):
    """The wrapper must call the inner train_step FIRST (so POET's merge has run)
    and only THEN read weights — i.e. it logs the post-step weight values."""
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
        Histogram=lambda x: ("HIST", len(x)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    # one selected layer-0 qkv weight; inner step "merges" by scaling it to 5.0
    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = torch.ones(2, 2) * 5.0  # simulate the post-step / post-merge value
        return ("loss", "skipped", "grad", "extra")

    opts = types.SimpleNamespace(
        log_weight_norms=True,
        log_weight_norms_interval=10,
        weight_norm_layers="0",
        num_layers=1,
        poet=False,
        poet_merge_period=0,
    )

    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    # train_step positional args: (..., model=args[2], ..., iteration=args[7])
    ret = wrapped(None, None, model, None, None, None, None, 10)

    assert ret == ("loss", "skipped", "grad", "extra")  # pass-through unchanged
    assert captured["_step"] == 10
    # row norm of a [5,5] row = sqrt(50); reads the POST-step weight, not the pre-step ones
    # (both rows equal, so mean == that row norm)
    assert captured["weightnorm/L0/qkv/row/mean"] == pytest.approx(50.0**0.5)


def test_wrapper_is_noop_when_flag_off(monkeypatch):
    import types

    calls = {"n": 0}

    def orig_train_step(*a, **k):
        calls["n"] += 1
        return "ret"

    opts = types.SimpleNamespace(log_weight_norms=False)
    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    assert wrapped(None, None, None) == "ret"
    assert calls["n"] == 1  # inner still runs; logging skipped


def test_patch_registers_with_empty_targets():
    import importlib

    from src.patches import registered_patches

    importlib.import_module("src.patches.weight_norm_monitor")
    reg = registered_patches()
    assert "weight_norm_monitor" in reg
    # targets=() so it composes with poet_merge_step's train_step wrapper
    assert reg["weight_norm_monitor"].targets == ()


def test_weight_norm_monitor_in_always_on_and_sorts_after_merge():
    from launchers.pretrain_gpt_slm import _ALWAYS_ON_PATCHES

    assert "weight_norm_monitor" in _ALWAYS_ON_PATCHES
    # registry applies in sorted order; outer wrapper must sort AFTER poet_merge_step
    assert "weight_norm_monitor" > "poet_merge_step"


def test_wrapper_warns_once_on_poet_cadence_misalignment(monkeypatch, caplog):
    import logging
    import sys
    import types

    import torch

    # wandb is present but iter 5 is not a merge boundary, so should_log is False
    # and the logging path is never reached — we are only exercising the warning.
    fake_wandb = types.SimpleNamespace(
        run=object(), log=lambda d, step=None: None, Histogram=lambda x: ("HIST", len(x))
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*a, **k):
        return "ret"

    # interval (5) is NOT a multiple of merge_period (400): effective cadence is
    # lcm(5, 400) = 400, far sparser than the requested 5 -> warn (once).
    opts = types.SimpleNamespace(
        log_weight_norms=True,
        log_weight_norms_interval=5,
        weight_norm_layers="0",
        num_layers=1,
        poet=True,
        poet_merge_period=400,
    )

    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    with caplog.at_level(logging.WARNING):
        wrapped(None, None, model, None, None, None, None, 5)
        wrapped(None, None, model, None, None, None, None, 10)  # second call: no new warning

    cadence = [r for r in caplog.records if "is not a multiple of" in r.getMessage()]
    assert len(cadence) == 1  # warn-once
    assert "400" in cadence[0].getMessage()  # reports the effective lcm cadence


def test_wrapper_no_cadence_warning_when_interval_is_multiple_of_merge_period(monkeypatch, caplog):
    import logging
    import sys
    import types

    import torch

    fake_wandb = types.SimpleNamespace(
        run=object(), log=lambda d, step=None: None, Histogram=lambda x: ("HIST", len(x))
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*a, **k):
        return "ret"

    # interval (800) IS a multiple of merge_period (400) -> aligned -> no warning
    opts = types.SimpleNamespace(
        log_weight_norms=True,
        log_weight_norms_interval=800,
        weight_norm_layers="0",
        num_layers=1,
        poet=True,
        poet_merge_period=400,
    )

    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    with caplog.at_level(logging.WARNING):
        wrapped(None, None, model, None, None, None, None, 400)

    assert not [r for r in caplog.records if "is not a multiple of" in r.getMessage()]
