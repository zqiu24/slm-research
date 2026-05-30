"""Warmup-Stable-Decay LR multiplier, matching src/utils/scheduler.py WSD.

Step-based (torchtitan drives the scheduler per optimizer step). Linear warmup
to 1.0 over `warmup_steps`, flat 1.0 plateau, then a linear decay over the final
`decay_steps` down to `floor`.

This is the FALLBACK path: torchtitan's native [lr_scheduler] already expresses
WSD (warmup_steps + decay_ratio + decay_type + min_lr_factor), so the primary
path is config-only (src/utils/torchtitan_args.lr_scheduler_block). This lambda
is kept + unit-tested so a custom LambdaLR builder is ready if a torchtitan bump
ever changes the native curve.
"""

from __future__ import annotations


def wsd_lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, decay_steps: int, floor: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return 1.0
    if decay_steps <= 0:
        return 1.0
    progress = min(1.0, (step - decay_start) / decay_steps)
    return 1.0 - (1.0 - floor) * progress
