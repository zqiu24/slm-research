"""Unit tests for src/utils/scheduler.py (pure config -> Megatron flags)."""

from __future__ import annotations

from src.utils.scheduler import scheduler_args


def _to_map(args: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


# total_tokens=6_000_000_000, seq_length=4096 -> 1_464_843 train samples.
_TOTAL = 6_000_000_000
_SEQ = 4096
_PEAK = 1.0e-3


def test_cosine_emits_decay_style_warmup_fraction_and_min_lr():
    m = _to_map(
        scheduler_args(
            {"type": "cosine", "warmup_fraction": 0.01, "min_lr_ratio": 0.1},
            peak_lr=_PEAK,
            total_tokens=_TOTAL,
            seq_length=_SEQ,
        )
    )
    assert m["--lr-decay-style"] == "cosine"
    assert m["--lr-warmup-fraction"] == "0.01"
    assert m["--min-lr"] == str(1.0e-3 * 0.1)


def test_constant_has_no_min_lr_and_no_decay_tail():
    m = _to_map(
        scheduler_args(
            {"type": "constant", "warmup_fraction": 0.01},
            peak_lr=_PEAK,
            total_tokens=_TOTAL,
            seq_length=_SEQ,
        )
    )
    assert m["--lr-decay-style"] == "constant"
    assert "--min-lr" not in m
    assert "--lr-wsd-decay-samples" not in m


def test_wsd_emits_wsd_style_and_tail_samples():
    m = _to_map(
        scheduler_args(
            {
                "type": "wsd",
                "warmup_fraction": 0.01,
                "min_lr_ratio": 0.1,
                "wsd_decay_fraction": 0.2,
                "wsd_decay_style": "cosine",
            },
            peak_lr=_PEAK,
            total_tokens=_TOTAL,
            seq_length=_SEQ,
        )
    )
    assert m["--lr-decay-style"] == "WSD"
    assert m["--lr-wsd-decay-style"] == "cosine"
    # round(0.2 * 6e9) // 4096
    assert m["--lr-wsd-decay-samples"] == str(int(round(0.2 * _TOTAL)) // _SEQ)


def test_inverse_square_root_type_is_normalized():
    m = _to_map(
        scheduler_args(
            {"type": "inverse_square_root", "warmup_fraction": 0.01, "min_lr_ratio": 0.1},
            peak_lr=_PEAK,
            total_tokens=_TOTAL,
            seq_length=_SEQ,
        )
    )
    assert m["--lr-decay-style"] == "inverse-square-root"


def test_warmup_tokens_converts_to_samples():
    m = _to_map(
        scheduler_args(
            {"type": "cosine", "warmup_tokens": 60_000_000, "min_lr_ratio": 0.1},
            peak_lr=_PEAK,
            total_tokens=_TOTAL,
            seq_length=_SEQ,
        )
    )
    assert m["--lr-warmup-samples"] == str(60_000_000 // _SEQ)
    assert "--lr-warmup-fraction" not in m
