"""Shared active-side signal for true single-side alternating POETX.

ONE source of truth — the current training iteration — read by the layer's
forward (which side to differentiate), the optimizer's step (which side's
momentum to advance + write), and the merge (which side to fold). Seeded once
per training step from Megatron's iteration (by the poet_merge_step train_step
wrapper) so all three agree within a step and resume keeps correct parity.

active_side convention (matches the optimizer's documented schedule):
    "out" on even cycles, "in" on odd, cycle length = alternate_every.
"""
from __future__ import annotations

_ITERATION = 0
_FIXED_SIDE = None  # None = alternate by iteration; "in"/"out" = pin one side


def set_iteration(it: int) -> None:
    global _ITERATION
    _ITERATION = int(it)


def get_iteration() -> int:
    return _ITERATION


def set_fixed_side(side) -> None:
    """Pin active_side() to one rotation side for the whole run (None = alternate).

    Set once at apply time from optim.poet.single_step_x_one_sided. Read by the
    optimizer write side (true_single_side) and the merge fold side via
    active_side(), so the one-sided POET mode stays self-consistent without
    touching optimizer/merge algorithms.
    """
    global _FIXED_SIDE
    if side not in (None, "in", "out"):
        raise ValueError(f"fixed_side must be None, 'in', or 'out', got {side!r}")
    _FIXED_SIDE = side


def active_side(alternate_every: int = 1) -> str:
    if _FIXED_SIDE is not None:
        return _FIXED_SIDE
    every = alternate_every if alternate_every and alternate_every > 0 else 1
    return "out" if (_ITERATION // every) % 2 == 0 else "in"
