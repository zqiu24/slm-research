from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_torchtitan import wandb_env_for
from src.utils.wandb_naming import wandb_run_name


def test_wandb_env_matches_run_identity():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",  # pin scale so the literal name below is deterministic
            "experiment=optim/adam",
            "backend=torchtitan",
            "wandb.project=pretrain-ablations-300m",
            # Permit a dirty tree: resolve_config stamps git_sha and would raise
            # GitDirty on a shared (non-slm-*) project name otherwise. This test
            # only checks W&B identity mapping, not git cleanliness.
            "allow_dirty=true",
        ]
    )
    resolve_config(cfg)
    env = wandb_env_for(cfg)
    assert env["WANDB_PROJECT"] == "pretrain-ablations-300m"
    # WANDB_NAME is the shared canonical name with the backend prefix — NOT the
    # timestamped run-dir name. Differs from the Megatron path only by "[torchtitan]".
    assert env["WANDB_NAME"] == wandb_run_name(cfg)
    assert env["WANDB_NAME"] == "[torchtitan] adam-llama3-300m-lr0.001"
