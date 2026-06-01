"""Unit tests for src/utils/param_count.py (pure trainable/total param counting).

CPU-only, no torch.distributed, no Megatron. Exercises the arithmetic the
wandb_trainable_params patch reduces across ranks.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.param_count import count_local_params


def _module(*param_specs: tuple[int, bool]) -> nn.Module:
    """Build a module holding params with the given (numel, requires_grad)."""
    m = nn.Module()
    for i, (numel, requires_grad) in enumerate(param_specs):
        m.register_parameter(f"p{i}", nn.Parameter(torch.zeros(numel), requires_grad=requires_grad))
    return m


def test_all_trainable_single_chunk():
    m = _module((10, True), (5, True))
    assert count_local_params([m]) == (15, 15)


def test_some_frozen_is_poet_like():
    # 100-element frozen base weight + 4-element trainable oft_R.
    m = _module((100, False), (4, True))
    assert count_local_params([m]) == (4, 104)


def test_all_frozen_has_zero_trainable():
    m = _module((50, False))
    assert count_local_params([m]) == (0, 50)


def test_sums_across_model_chunks():
    a = _module((10, True), (90, False))
    b = _module((6, True))
    assert count_local_params([a, b]) == (16, 106)


def test_empty_chunk_list():
    assert count_local_params([]) == (0, 0)
