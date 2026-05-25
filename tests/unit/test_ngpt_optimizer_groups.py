"""CPU tests for the nGPT param-group classifier."""

import torch.nn as nn

from src.model.ngpt.scaling_params import LearnedScaling
from src.patches.ngpt_optimizer_setup import classify_ngpt_param_groups


def test_scaling_params_get_zero_wd_group():
    m = nn.Module()
    m.linear = nn.Linear(8, 8, bias=False)
    m.attn_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.mlp_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.sqk = LearnedScaling((8,), init_value=1.0, init_scaling=1.0 / 2.83)
    m.suv = LearnedScaling((8,), init_value=1.0, init_scaling=1.0)
    m._ngpt_sz = LearnedScaling((100,), init_value=1.0, init_scaling=1.0 / 2.83)

    decay, no_decay = classify_ngpt_param_groups(m)
    # linear.weight should be in decay (it is the only matrix param)
    decay_ids = {id(p) for p in decay}
    no_decay_ids = {id(p) for p in no_decay}
    assert id(m.linear.weight) in decay_ids
    for p in (m.attn_alpha.param, m.mlp_alpha.param, m.sqk.param, m.suv.param, m._ngpt_sz.param):
        assert (
            id(p) in no_decay_ids
        ), f"scaling param shape {p.shape} should be in the no-decay group"
