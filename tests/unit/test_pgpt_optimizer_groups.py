"""pgpt scaling-param classifier (no-WD group membership)."""

import torch.nn as nn

from src.model.pgpt.scaling_params import LearnedScaling
from src.patches.pgpt_optimizer_setup import classify_pgpt_scaling_params


def test_scaling_params_are_classified():
    m = nn.Module()
    m.linear = nn.Linear(8, 8, bias=False)
    m.attn_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.mlp_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.sqk = LearnedScaling((8,), init_value=1.0, init_scaling=1.0 / 2.83)
    m.suv = LearnedScaling((8,), init_value=1.0, init_scaling=1.0)
    m._ngpt_sz = LearnedScaling((100,), init_value=1.0, init_scaling=1.0 / 2.83)

    ids = {id(p) for p in classify_pgpt_scaling_params(m)}
    assert id(m.linear.weight) not in ids
    for p in (m.attn_alpha.param, m.mlp_alpha.param, m.sqk.param, m.suv.param, m._ngpt_sz.param):
        assert id(p) in ids
