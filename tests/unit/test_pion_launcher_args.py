# tests/unit/test_pion_launcher_args.py
"""The launcher's add_slm_args registers Pion flags + the pion slm-optimizer choice."""

from __future__ import annotations

import argparse

from launchers.pretrain_gpt_slm import add_slm_args


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    return parser.parse_args(["--slm-config-path", "x", *argv])


def test_pion_is_a_valid_slm_optimizer_choice():
    args = _parse(["--slm-optimizer", "pion"])
    assert args.slm_optimizer == "pion"


def test_pion_args_have_reference_defaults():
    args = _parse([])
    assert args.pion_scaling == "rms"
    assert args.pion_rms == 0.2
    assert args.pion_update_side == "both"
    assert args.pion_momentum == "transported_ambient_ambient"
    assert args.pion_degree == 2
    assert args.pion_beta1 == 0.9
    assert args.pion_beta2 == 0.999
    assert args.pion_use_second_momentum is False


def test_pion_args_are_overridable():
    args = _parse(
        [
            "--pion-update-side",
            "alternate",
            "--pion-momentum",
            "lie_lie",
            "--pion-beta2",
            "0.95",
            "--pion-use-second-momentum",
        ]
    )
    assert args.pion_update_side == "alternate"
    assert args.pion_momentum == "lie_lie"
    assert args.pion_beta2 == 0.95
    assert args.pion_use_second_momentum is True
