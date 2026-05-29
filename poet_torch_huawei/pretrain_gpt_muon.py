# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Muon pretraining entry point for GPT.

Muon integration is handled entirely via `--optimizer muon` and the
`--muon-*` CLI flags (wired into `OptimizerConfig`), plus `ns_shape`
attributes set on MLA up-projection weights in
`megatron.core.transformer.multi_latent_attention` when
`--muon-ns-num-head-groups > 0`. No training-loop changes are required.

This file is a thin shim over `pretrain_gpt.py` that re-uses the same
providers and forward step, but exists as a dedicated entry point so that
launch scripts can unambiguously opt into the Muon training recipe.
"""

from megatron.core.enums import ModelType
from megatron.training import inprocess_restart

from pretrain_gpt import (
    forward_step,
    model_provider,
    train_valid_test_datasets_provider,
)

try:
    from megatron.post_training.arguments import add_modelopt_args

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False


if __name__ == "__main__":

    from megatron.training import pretrain

    # Match `pretrain_gpt.py`: tell the core dataset builder that this
    # provider is distributed-aware.
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable in-process restart before entering `pretrain`.
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
