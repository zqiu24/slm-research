from __future__ import annotations

import math

from src.titan_ext.lr_scheduler import wsd_lr_multiplier
from src.utils.torchtitan_args import lr_scheduler_block


def test_warmup_then_stable_then_decay_to_floor():
    total = 1000
    warmup = 100
    decay = 200  # last 200 steps decay
    floor = 0.1

    # warmup ramp
    assert math.isclose(wsd_lr_multiplier(0, total, warmup, decay, floor), 0.0, abs_tol=1e-6)
    assert math.isclose(wsd_lr_multiplier(warmup, total, warmup, decay, floor), 1.0, abs_tol=1e-6)
    # stable plateau
    assert math.isclose(wsd_lr_multiplier(500, total, warmup, decay, floor), 1.0, abs_tol=1e-6)
    # end of run decays to the floor
    assert math.isclose(wsd_lr_multiplier(total, total, warmup, decay, floor), floor, abs_tol=1e-6)


def test_lr_scheduler_block_maps_to_torchtitan_keys():
    sched = {
        "type": "wsd",
        "warmup_fraction": 0.1,
        "wsd_decay_fraction": 0.2,
        "wsd_decay_style": "cosine",
        "min_lr_ratio": 0.1,
    }
    block = lr_scheduler_block(sched, total_steps=1000)
    assert block["warmup_steps"] == 100
    assert math.isclose(block["decay_ratio"], 0.2)  # fraction, not a step count
    assert block["decay_type"] == "cosine"
    assert math.isclose(block["min_lr_factor"], 0.1)
