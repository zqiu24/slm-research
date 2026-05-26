"""Tests for slm-research's Megatron GPT entrypoint helpers."""

from __future__ import annotations

import argparse

from launchers.pretrain_gpt_slm import add_slm_args


def test_add_slm_args_accepts_poet_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)

    args = parser.parse_args(
        [
            "--slm-config-path",
            "runs/champion-llama3-1_2b-s42-20260524T200332Z/resolved_config.yaml",
            "--slm-optimizer",
            "poet",
            "--poet",
            "--poet-block-size",
            "256",
            "--poet-init-type",
            "normalized",
            "--poet-mup-alpha",
            "1.0",
            "--poet-merge-period",
            "200",
            "--poet-scale",
            "1.5",
        ]
    )

    assert (
        args.slm_config_path
        == "runs/champion-llama3-1_2b-s42-20260524T200332Z/resolved_config.yaml"
    )
    assert args.slm_optimizer == "poet"
    assert args.poet is True
    assert args.poet_block_size == 256
    assert args.poet_merge_period == 200
    assert args.poet_scale == 1.5


def test_add_slm_args_accepts_step_decay_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)

    args = parser.parse_args(
        [
            "--slm-config-path",
            "runs/_ignored/resolved_config.yaml",
            "--lr-decay-step-ratio",
            "0.8",
            "0.9",
            "--lr-decay-step-coeff",
            "0.316",
            "0.1",
        ]
    )
    assert args.lr_decay_step_ratio == [0.8, 0.9]
    assert args.lr_decay_step_coeff == [0.316, 0.1]


def test_add_slm_args_step_decay_default_is_none():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)

    args = parser.parse_args(["--slm-config-path", "runs/_ignored/resolved_config.yaml"])
    assert args.lr_decay_step_ratio is None
    assert args.lr_decay_step_coeff is None
