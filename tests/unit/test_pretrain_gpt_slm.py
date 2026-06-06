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


def test_add_slm_args_accepts_unfuse_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--unfuse-qkv",
            "--unfuse-fc1",
        ]
    )
    assert args.unfuse_qkv is True
    assert args.unfuse_fc1 is True


def test_unfuse_flags_default_false():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert args.unfuse_qkv is False
    assert args.unfuse_fc1 is False


def test_add_slm_args_accepts_poet_reinit_period():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet",
            "--poet-merge-period",
            "1",
            "--poet-reinit-period",
            "400",
        ]
    )
    assert args.poet_merge_period == 1
    assert args.poet_reinit_period == 400


def test_add_slm_args_poet_reinit_period_defaults_zero():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_reinit_period == 0


def test_add_slm_args_accepts_lie_algebra_args():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet",
            "--poet-q-optimizer",
            "lie_algebra",
            "--poet-lie-b1",
            "0.9",
            "--poet-lie-b2",
            "0.95",
            "--poet-lie-eps",
            "1e-8",
            "--poet-lie-v-mode",
            "elementwise",
        ]
    )
    assert args.poet_q_optimizer == "lie_algebra"
    assert args.poet_lie_b1 == 0.9 and args.poet_lie_b2 == 0.95
    assert args.poet_lie_eps == 1e-8 and args.poet_lie_v_mode == "elementwise"


def test_add_slm_args_lie_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_lie_b1 == 0.9 and args.poet_lie_b2 == 0.95
    assert args.poet_lie_eps == 1e-8 and args.poet_lie_v_mode == "elementwise"


def test_add_slm_args_accepts_lie_alternating():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet",
            "--poet-lie-alternating",
            "--poet-lie-alternate-every",
            "2",
        ]
    )
    assert args.poet_lie_alternating is True
    assert args.poet_lie_alternate_every == 2


def test_add_slm_args_lie_alternating_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_lie_alternating is False
    assert args.poet_lie_alternate_every == 1


def test_add_slm_args_accepts_lie_rms():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--poet", "--poet-lie-rms", "--poet-lie-rms-c", "0.3"]
    )
    assert args.poet_lie_rms is True
    assert args.poet_lie_rms_c == 0.3


def test_add_slm_args_lie_rms_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_lie_rms is False
    assert args.poet_lie_rms_c == 0.2


def test_add_slm_args_accepts_head_aligned_flags():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet",
            "--poet-head-aligned-attn",
            "--poet-no-head-resid-perm",
        ]
    )
    assert args.poet_head_aligned_attn is True
    assert args.poet_no_head_resid_perm is True


def test_add_slm_args_head_aligned_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_head_aligned_attn is False
    assert args.poet_no_head_resid_perm is False


def test_add_slm_args_accepts_lie_ortho():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet-q-optimizer",
            "lie_ortho",
            "--poet-lie-ortho-c",
            "0.02",
            "--poet-lie-ortho-method",
            "spectral",
            "--poet-lie-ortho-ns-steps",
            "20",
            "--poet-lie-ortho-use-second-moment",
        ]
    )
    assert args.poet_q_optimizer == "lie_ortho"
    assert args.poet_lie_ortho_c == 0.02
    assert args.poet_lie_ortho_method == "spectral"
    assert args.poet_lie_ortho_ns_steps == 20
    assert args.poet_lie_ortho_use_second_moment is True


def test_lie_ortho_knobs_default():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert args.poet_lie_ortho_c == 0.01
    assert args.poet_lie_ortho_method == "muon"
    assert args.poet_lie_ortho_ns_steps == 5
    assert args.poet_lie_ortho_use_second_moment is False


def test_add_slm_args_accepts_lie_ortho_distributed():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    on = parser.parse_args(["--slm-config-path", "x.yaml", "--poet-lie-ortho-distributed"])
    off = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert on.poet_lie_ortho_distributed is True
    assert off.poet_lie_ortho_distributed is False
