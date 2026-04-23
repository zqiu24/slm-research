"""Tests for parallelism resolution from cluster rules."""

from __future__ import annotations

import pytest

from src.utils.parallelism import resolve


H800_CN_RULES = [
    {"model_params_lt": 3.0e9, "tp": 1, "pp": 1},
    {"model_params_lt": 1.0e10, "tp": 4, "pp": 1},
    {"model_params_lt": 3.0e10, "tp": 8, "pp": 1},
]


def test_small_model_no_tp():
    p = resolve(1_200_000_000, cluster_nodes=6, gpus_per_node=8, tp_size_rules=H800_CN_RULES)
    assert (p.tp, p.pp, p.dp) == (1, 1, 48)


def test_7b_gets_tp4():
    p = resolve(7_000_000_000, cluster_nodes=6, gpus_per_node=8, tp_size_rules=H800_CN_RULES)
    assert (p.tp, p.pp) == (4, 1)
    assert p.dp == 48 // 4


def test_world_size_divisibility_enforced():
    with pytest.raises(ValueError, match="not divisible"):
        resolve(
            7_000_000_000, cluster_nodes=1, gpus_per_node=6, tp_size_rules=H800_CN_RULES
        )


def test_exceeding_all_rules_uses_last():
    p = resolve(
        1_000_000_000_000, cluster_nodes=8, gpus_per_node=8, tp_size_rules=H800_CN_RULES
    )
    assert p.tp == 8 and p.pp == 1
