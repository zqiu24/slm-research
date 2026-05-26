"""Step-decay LR schedule wiring through megatron_args.py."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from launchers.submit import _parse_overrides
from src.utils.megatron_args import _training_args, build_megatron_args


def _flatten(args: list[str]) -> dict[str, list[str]]:
    """Group consecutive non-flag tokens under their preceding flag."""
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(args):
        key = args[i]
        i += 1
        values: list[str] = []
        while i < len(args) and not args[i].startswith("--"):
            values.append(args[i])
            i += 1
        out[key] = values
    return out


def _minimal_cfg(**overrides):
    base = OmegaConf.create(
        {
            "training": {
                "global_batch_size_tokens": 4_194_304,
                "tokens_per_param": 20,
            },
            "optim": {"lr": 9.0e-4, "weight_decay": 0.1},
            "base": {
                "model": {"seq_length": 4096},
                "non_embedding_params": 520_000_000,
            },
        }
    )
    # OmegaConf.merge needs DictConfig on both sides
    return OmegaConf.merge(base, OmegaConf.create(overrides))


def test_step_decay_emits_ratio_and_coeff_flags():
    cfg = _minimal_cfg(
        training={
            "lr_decay_style": "step",
            "lr_decay_step_ratio": [0.8, 0.9],
            "lr_decay_step_coeff": [0.316, 0.1],
        }
    )
    args = _training_args(cfg)
    flat = _flatten(args)

    assert flat["--lr-decay-style"] == ["step"]
    assert flat["--lr-decay-step-ratio"] == ["0.8", "0.9"]
    assert flat["--lr-decay-step-coeff"] == ["0.316", "0.1"]


def test_cosine_default_does_not_emit_step_decay_flags():
    cfg = _minimal_cfg()
    args = _training_args(cfg)

    assert "--lr-decay-step-ratio" not in args
    assert "--lr-decay-step-coeff" not in args


def test_step_decay_without_ratio_raises():
    cfg = _minimal_cfg(training={"lr_decay_style": "step"})
    with pytest.raises(ValueError, match="lr_decay_step_ratio"):
        _training_args(cfg)


def test_adamw_step_decay_experiment_resolves_and_emits():
    # End-to-end through Hydra composition: the new experiment file should
    # produce a torchrun command containing the step-decay flags.
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_3b",
            "experiment=optim/adamw_step_decay",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    args = build_megatron_args(cfg)
    flat = _flatten(args)

    assert flat["--lr-decay-style"] == ["step"]
    assert flat["--lr-decay-step-ratio"] == ["0.8", "0.9"]
    assert flat["--lr-decay-step-coeff"] == ["0.316", "0.1"]
    assert flat["--lr"] == ["0.0009"]
