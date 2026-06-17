"""Guard: LieOrthMomentum assumes each oft_R param is 2-D (n_blocks, n_elems) and
batches across params. GroupedPOETXLinear must therefore keep E separate 2-D oft_R
params, never a stacked 3-D param. If this test breaks, the grouped design's
optimizer-compatibility assumption broke with it."""

import torch

from src.optim.poet_lie_orth import LieOrthMomentum


def test_lie_ortho_steps_2d_oft_and_rejects_3d():
    # block_size 4 -> n_elems = 4*3/2 = 6 ; two blocks
    p2d = torch.nn.Parameter(torch.zeros(2, 6))  # (n_blocks, n_elems)
    p2d.grad = torch.randn(2, 6)
    opt = LieOrthMomentum(
        [{"params": [p2d], "use_skew": True, "side": "in", "lr": 1e-2}],
        ortho_c=8,
        ortho_method="muon",
        ortho_ns_steps=5,
    )
    opt.step()  # 2-D path works
    assert torch.isfinite(p2d).all() and p2d.abs().sum() > 0

    # A stacked 3-D oft_R would mis-read n_elems from the wrong axis -> the optimizer
    # cannot consume it. We assert the 2-D contract here so the grouped module honors it.
    p3d = torch.nn.Parameter(torch.zeros(3, 2, 6))  # (E, n_blocks, n_elems)
    p3d.grad = torch.randn(3, 2, 6)
    opt3 = LieOrthMomentum(
        [{"params": [p3d], "use_skew": True, "side": "in", "lr": 1e-2}],
        ortho_c=8,
        ortho_method="muon",
        ortho_ns_steps=5,
    )
    with __import__("pytest").raises(Exception):
        opt3.step()  # 3-D oft_R is unsupported
