"""Per-rank Megatron Mamba/hybrid entrypoint for slm-research.

Mirror of pretrain_gpt_slm.py for families whose layer stack contains Mamba2
layers (``base.model.entrypoint: mamba``, e.g. nemotron_h). Megatron builds
these through MambaModel (pretrain_mamba.py + mamba_builders.py at the pin
root), not GPTModel — the GPT entrypoint cannot express the M/*/- pattern.

Deliberate differences from the GPT twin:
- imports pretrain_mamba / mamba_builder instead of pretrain_gpt / gpt_builder
- no titan_init wrapping (a GPT-path reproduction concern)
- no get_embedding_ranks kwarg (pretrain_mamba.py does not pass one)

NOTE: this path has no MTP support (pretrain_gpt.py docstring, pin
core_v0.17.0); src/utils/megatron_args.py rejects mtp on hybrid configs.
The always-on patches compose unchanged: they target
megatron.training.training symbols shared by both entrypoints (the
pretrain_gpt-targeted ones like overfit_single_batch import cleanly and are
simply never hit on this path).
"""

from __future__ import annotations

import os
import sys
from functools import partial

from launchers.pretrain_gpt_slm import (
    _apply_runtime_patches,
    _combined_extra_args_provider,
    _load_resolved_config,
    _prepend_paths,
)


def main() -> None:
    _prepend_paths()

    config_path = None
    for idx, item in enumerate(sys.argv):
        if item == "--slm-config-path" and idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
            break
    if config_path is None:
        raise RuntimeError("--slm-config-path must be present in torchrun args")

    cfg = _load_resolved_config(config_path)

    if bool(cfg.get("cluster", {}).get("wandb_offline", False)):
        os.environ.setdefault("WANDB_MODE", "offline")

    _apply_runtime_patches(cfg)

    import pretrain_mamba as mm
    from mamba_builders import mamba_builder
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps

    set_startup_timestamps(
        program_start=mm._PROGRAM_START_TIME,
        main_entry=mm.time.time(),
    )
    mm.train_valid_test_datasets_provider.is_distributed = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        mm.train_valid_test_datasets_provider,
        partial(mm.model_provider, mamba_builder),
        ModelType.encoder_or_decoder,
        mm.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=_combined_extra_args_provider(
            mm.add_modelopt_args if mm.has_nvidia_modelopt else None
        ),
        store=store,
    )


if __name__ == "__main__":
    main()
