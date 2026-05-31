"""The unified W&B run-name builder shared by both training backends.

Contract: both backends produce the SAME canonical name (`wandb_base_name`); the
only difference is the `[megatron]` / `[torchtitan]` prefix added by
`wandb_run_name`.
"""

from __future__ import annotations

from launchers.submit import _parse_overrides
from src.utils.wandb_naming import wandb_base_name, wandb_run_name


def _cfg(*overrides):
    return _parse_overrides(["base/family=llama3", "base/scale=300m", *overrides])


def test_base_name_is_backend_independent():
    # champion is adam, lr 1e-3 — no seed, no timestamp, no backend segment.
    assert wandb_base_name(_cfg("experiment=champion")) == "adam-llama3-300m-lr0.001"


def test_run_name_default_backend_is_megatron():
    # backend unset → defaults to megatron (matches resolve_config).
    assert wandb_run_name(_cfg("experiment=champion")) == "[megatron] adam-llama3-300m-lr0.001"


def test_run_name_torchtitan_prefix():
    assert (
        wandb_run_name(_cfg("experiment=champion", "backend=torchtitan"))
        == "[torchtitan] adam-llama3-300m-lr0.001"
    )


def test_poet_name_order_is_block_lr_scale():
    # POET segment order: <exp>-<family>-<scale>-<block>-lr<lr>-scale<v> (block
    # BEFORE lr). Explicit overrides keep this robust to the yaml defaults.
    name = wandb_base_name(
        _cfg(
            "experiment=optim/poet",
            "optim.lr=0.001",
            "optim.poet.block_count=4",
            "optim.poet.scale=0.05",
        )
    )
    assert name == "poet-llama3-300m-bc4-lr0.001-scale0.05"


def test_poet_scale_formatting_is_compact():
    # `:g` keeps it tidy: 0.05 not 0.050000, 1.0 -> 1.
    assert wandb_base_name(_cfg("experiment=optim/poet", "optim.poet.scale=0.05")).endswith(
        "-scale0.05"
    )
    assert wandb_base_name(_cfg("experiment=optim/poet", "optim.poet.scale=1.0")).endswith(
        "-scale1"
    )


def test_only_difference_between_backends_is_the_prefix():
    base = _cfg("experiment=champion")
    titan = _cfg("experiment=champion", "backend=torchtitan")
    # Identical canonical bodies; names differ ONLY by the bracketed prefix.
    assert wandb_base_name(base) == wandb_base_name(titan)
    assert wandb_run_name(base) == f"[megatron] {wandb_base_name(base)}"
    assert wandb_run_name(titan) == f"[torchtitan] {wandb_base_name(titan)}"
    assert wandb_run_name(base).removeprefix("[megatron] ") == wandb_run_name(titan).removeprefix(
        "[torchtitan] "
    )
