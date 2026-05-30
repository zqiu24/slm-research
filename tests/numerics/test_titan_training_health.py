"""M3 gate: a torchtitan run trains and the loss decreases healthily.

Marked gpu + slow + numerics. NOT run in the dev harness — the operator runs the
runbook (docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md) on a
node, then points TT_LOSS_LOG at the resulting per-step loss jsonl. Asserts: no
NaN/Inf, and the final-window mean loss is meaningfully below the early-window
mean (the model is learning). Thresholds are deliberately loose for M1; tighten
once real curves are in hand.
"""

from __future__ import annotations

import json
import math
import os

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.numerics]


def _losses(path):
    with open(path) as fh:
        return [json.loads(line)["loss"] for line in fh]


def test_torchtitan_run_is_healthy():
    log = os.environ.get("TT_LOSS_LOG")
    if not log:
        pytest.skip("set TT_LOSS_LOG to a torchtitan run's per-step loss jsonl")
    losses = _losses(log)
    assert len(losses) >= 40, "need enough steps to judge the curve"
    assert all(math.isfinite(x) for x in losses), "loss has NaN/Inf"
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early - 0.1, f"loss not decreasing: early={early:.3f} late={late:.3f}"
