"""Resolve a `scheduler:` config block into Megatron LR/decay CLI flags.

Pure function — no Megatron import, no torch. Unit-testable on CPU.
See docs/superpowers/specs/2026-05-27-scheduler-config-axis-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# config type (normalized, dashes->underscores, lowercased) -> Megatron value
_DECAY_STYLE = {
    "constant": "constant",
    "linear": "linear",
    "cosine": "cosine",
    "inverse_square_root": "inverse-square-root",
    "wsd": "WSD",
}
_VALID_WSD_DECAY_STYLES = ("exponential", "linear", "cosine", "minus_sqrt")

_COMMON_FIELDS = {"type", "warmup_fraction", "warmup_tokens", "min_lr_ratio"}
# Extra fields each type may set, beyond the common ones.
_TYPE_EXTRA_FIELDS = {
    "constant": set(),
    "linear": set(),
    "cosine": set(),
    "inverse_square_root": set(),
    "wsd": {"wsd_decay_fraction", "wsd_decay_style"},
}


def _norm_type(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    if key not in _DECAY_STYLE:
        raise ValueError(
            f"unknown scheduler type {raw!r}; valid: "
            "constant, linear, cosine, inverse-square-root, wsd"
        )
    return key


def _warmup_args(sched: Mapping[str, Any], *, seq_length: int, force_no_warmup: bool) -> list[str]:
    has_frac = sched.get("warmup_fraction", None) is not None
    has_tok = sched.get("warmup_tokens", None) is not None
    if has_frac and has_tok:
        raise ValueError("set only one of warmup_fraction / warmup_tokens, not both")
    if force_no_warmup:
        return ["--lr-warmup-samples", "0"]
    if has_tok:
        wt = int(sched["warmup_tokens"])
        if wt < 0:
            raise ValueError(f"warmup_tokens must be >= 0, got {wt}")
        return ["--lr-warmup-samples", str(wt // seq_length)]
    wf = float(sched.get("warmup_fraction", 0.0))
    if not (0.0 <= wf < 1.0):
        raise ValueError(f"warmup_fraction must be in [0, 1), got {wf}")
    return ["--lr-warmup-fraction", str(wf)]


def scheduler_args(
    sched: Mapping[str, Any],
    *,
    peak_lr: float,
    total_tokens: int,
    seq_length: int,
    force_no_warmup: bool = False,
) -> list[str]:
    """Validate `sched` and return the Megatron LR/decay CLI flags.

    `peak_lr` is cfg.optim.lr (emitted separately as --lr by the caller);
    here it only scales min_lr_ratio. `force_no_warmup` lets the caller
    preserve nGPT's no-LR-warmup behavior regardless of the scheduler file.
    """
    if "type" not in sched:
        raise ValueError("scheduler block missing required field 'type'")
    norm = _norm_type(str(sched["type"]))

    allowed = _COMMON_FIELDS | _TYPE_EXTRA_FIELDS[norm]
    foreign = set(sched.keys()) - allowed
    if foreign:
        raise ValueError(
            f"scheduler type {norm!r} does not accept field(s) {sorted(foreign)}; "
            f"allowed: {sorted(allowed)}"
        )

    args: list[str] = ["--lr-decay-style", _DECAY_STYLE[norm]]

    if norm != "constant":
        min_lr_ratio = sched.get("min_lr_ratio", None)
        if min_lr_ratio is not None:
            args += ["--min-lr", str(float(min_lr_ratio) * float(peak_lr))]

    args += _warmup_args(sched, seq_length=seq_length, force_no_warmup=force_no_warmup)

    if norm == "wsd":
        wdf = sched.get("wsd_decay_fraction", None)
        if wdf is None:
            raise ValueError("scheduler type 'wsd' requires 'wsd_decay_fraction'")
        wdf = float(wdf)
        if not (0.0 < wdf <= 1.0):
            raise ValueError(f"wsd_decay_fraction must be in (0, 1], got {wdf}")
        wsd_decay_samples = int(round(wdf * total_tokens)) // seq_length
        args += ["--lr-wsd-decay-samples", str(wsd_decay_samples)]
        wsd_style = str(sched.get("wsd_decay_style", "cosine"))
        if wsd_style not in _VALID_WSD_DECAY_STYLES:
            raise ValueError(
                f"wsd_decay_style must be one of {list(_VALID_WSD_DECAY_STYLES)}, "
                f"got {wsd_style!r}"
            )
        args += ["--lr-wsd-decay-style", wsd_style]

    return args
