"""One-sided POETX layers: train exactly one fixed rotation side; the frozen
side's gradient is shape-correct zeros (never trained)."""

import pytest
import torch
from poet_torch import InOnlyPOETXLinear, OneSidedPOETXLinear, OutOnlyPOETXLinear


def _run_backward(pl):
    with torch.no_grad():
        pl.weight.normal_()
    x = torch.randn(4, pl.in_features, requires_grad=True)
    gy = torch.randn(4, pl.out_features)
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    (pl(x) * gy).sum().backward()
    return pl


def test_in_only_trains_in_freezes_out():
    pl = _run_backward(InOnlyPOETXLinear(in_features=12, out_features=8, block_count=1))
    assert pl.side == "in"
    assert pl.alternating is True
    assert pl.oft_R_in.grad.abs().sum() > 0
    assert torch.count_nonzero(pl.oft_R_out.grad) == 0


def test_out_only_trains_out_freezes_in():
    pl = _run_backward(OutOnlyPOETXLinear(in_features=12, out_features=8, block_count=1))
    assert pl.side == "out"
    assert pl.oft_R_out.grad.abs().sum() > 0
    assert torch.count_nonzero(pl.oft_R_in.grad) == 0


def test_one_sided_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        OneSidedPOETXLinear(in_features=12, out_features=8, block_count=1, side="left")
