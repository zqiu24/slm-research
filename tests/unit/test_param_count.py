"""Unit tests for src/utils/param_count.py (pure trainable/total/poet counting).

CPU-only, no torch.distributed, no Megatron. Exercises the arithmetic the
wandb_trainable_params patch reduces across ranks.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.param_count import count_local_params


def _module(*param_specs: tuple[int, bool]) -> nn.Module:
    """Build a module holding params with the given (numel, requires_grad).

    Params are named ``p0``, ``p1``, ... (no ``oft_R``), so ``poet`` is 0.
    """
    m = nn.Module()
    for i, (numel, requires_grad) in enumerate(param_specs):
        m.register_parameter(f"p{i}", nn.Parameter(torch.zeros(numel), requires_grad=requires_grad))
    return m


def _named_module(*param_specs: tuple[str, int, bool]) -> nn.Module:
    """Build a module holding params with explicit (name, numel, requires_grad)."""
    m = nn.Module()
    for name, numel, requires_grad in param_specs:
        m.register_parameter(name, nn.Parameter(torch.zeros(numel), requires_grad=requires_grad))
    return m


def test_all_trainable_single_chunk():
    m = _module((10, True), (5, True))
    assert count_local_params([m]) == (15, 15, 0)


def test_some_frozen_is_poet_like():
    # 100-element frozen base weight + 4-element trainable oft_R.
    m = _module((100, False), (4, True))
    assert count_local_params([m]) == (4, 104, 0)


def test_all_frozen_has_zero_trainable():
    m = _module((50, False))
    assert count_local_params([m]) == (0, 50, 0)


def test_sums_across_model_chunks():
    a = _module((10, True), (90, False))
    b = _module((6, True))
    assert count_local_params([a, b]) == (16, 106, 0)


def test_empty_chunk_list():
    assert count_local_params([]) == (0, 0, 0)


def test_poet_counts_oft_r_by_name():
    # 100 frozen base weight (not poet) + 4-element trainable oft_R (poet).
    m = _named_module(("base_weight", 100, False), ("oft_R", 4, True))
    assert count_local_params([m]) == (4, 104, 4)


def test_poet_counts_decoupled_in_and_out():
    # Decoupled layout exposes oft_R_in / oft_R_out; both count toward poet.
    m = _named_module(("oft_R_in", 3, True), ("oft_R_out", 5, True), ("norm", 7, True))
    assert count_local_params([m]) == (15, 15, 8)


def test_poet_is_zero_without_oft_r():
    # adam / muon / ngpt: no oft_R-named params -> poet is 0.
    m = _named_module(("weight", 20, True), ("bias", 4, True))
    assert count_local_params([m]) == (24, 24, 0)
