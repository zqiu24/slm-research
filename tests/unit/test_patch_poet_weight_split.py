"""Tests for the weight-split capture hook (src/patches/poet_weight_split_log.py):
the fwd/bwd hook must (a) never raise on a no-grad forward — diagnostics must not
crash training — and (b) accumulate G = g_y^T x on a real grad forward.
"""

import pytest
import torch
import torch.nn as nn

from src.patches.poet_weight_split_log import _capture, _install_capture_hook


@pytest.fixture(autouse=True)
def _capture_on():
    _capture["on"] = True
    yield
    _capture["on"] = False


class _Lin(nn.Module):
    def __init__(self, out_f=4, in_f=3):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_f, in_f))

    def forward(self, x):
        return x @ self.w.transpose(0, 1)  # (.., in) -> (.., out)


def test_capture_hook_does_not_raise_on_no_grad_forward():
    # An eval / init forward produces an output that does NOT require grad.
    # register_hook would raise on it; the hook must skip instead of crashing.
    m = _Lin()
    _install_capture_hook(m)
    x = torch.randn(5, 3)
    with torch.no_grad():
        m(x)  # must not raise
    assert getattr(m, "_coord_G", None) is None


def test_capture_hook_accumulates_g_on_grad_forward():
    m = _Lin(out_f=4, in_f=3)
    _install_capture_hook(m)
    x = torch.randn(5, 3)
    m(x).sum().backward()
    g = getattr(m, "_coord_G", None)
    assert g is not None
    assert g.shape == (4, 3)  # G = g_y^T x, (out, in)


def test_capture_hook_inert_when_capture_off():
    _capture["on"] = False
    m = _Lin()
    _install_capture_hook(m)
    m(torch.randn(5, 3)).sum().backward()
    assert getattr(m, "_coord_G", None) is None
