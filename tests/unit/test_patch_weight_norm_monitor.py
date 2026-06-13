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

    # row norms = [3, 4]; row_rms = norm / sqrt(in_dim=3)
    assert stats["row"]["min"] == pytest.approx(3.0)
    assert stats["row"]["max"] == pytest.approx(4.0)
    assert stats["row"]["mean"] == pytest.approx(3.5)
    assert stats["row_rms"]["max"] == pytest.approx(4.0 / math.sqrt(3))
    # col norms = [3, 4, 0]; col_rms = norm / sqrt(out_dim=2)
    assert stats["col"]["max"] == pytest.approx(4.0)
    assert stats["col"]["min"] == pytest.approx(0.0)
    assert stats["col_rms"]["max"] == pytest.approx(4.0 / math.sqrt(2))
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

    # scalar keys for both matrices in layer 0, all four kinds
    assert "weightnorm/L0/qkv/row/mean" in captured
    assert "weightnorm/L0/qkv/col_rms/max" in captured
    assert "weightnorm/L0/fc1/row/mean" in captured
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

    fake_wandb = types.SimpleNamespace(run=None, log=lambda d, step=None: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    opts = types.SimpleNamespace(num_layers=2, weight_norm_layers="0")
    from src.patches.weight_norm_monitor import _log_weight_norms

    # must not raise even though no run is active
    _log_weight_norms(_fake_model(), iteration=100, opts=opts)
