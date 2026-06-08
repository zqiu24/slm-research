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


def set_iteration(it: int) -> None:
    global _ITERATION
    _ITERATION = int(it)


def get_iteration() -> int:
    return _ITERATION


def active_side(alternate_every: int = 1) -> str:
    every = alternate_every if alternate_every and alternate_every > 0 else 1
    return "out" if (_ITERATION // every) % 2 == 0 else "in"
