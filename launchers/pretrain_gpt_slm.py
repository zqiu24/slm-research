"""Per-rank Megatron GPT entrypoint for slm-research.

This module is launched by torchrun. It applies slm-research patches inside
the rank process, then calls Megatron's GPT pretrain function from the pinned
third_party checkout.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGATRON_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"


def add_slm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("slm-research")
    group.add_argument("--slm-config-hash", type=str, required=True)
    group.add_argument("--slm-optimizer", choices=["adamw", "muon", "poet"], default="adamw")
    group.add_argument("--poet", action="store_true")
    group.add_argument("--poet-block-size", type=int, default=256)
    group.add_argument(
        "--poet-init-type",
        choices=["none", "normalized", "mup_normalized"],
        default="normalized",
    )
    group.add_argument("--poet-mup-alpha", type=float, default=1.0)
    group.add_argument("--poet-merge-period", type=int, default=0)
    group.add_argument("--poet-scale", type=float, default=1.0)
    group.add_argument(
        "--poet-cache-mode",
        choices=["none", "cached_fwd", "cached_fwd_bwd"],
        default="none",
    )
    return parser


def _prepend_paths() -> None:
    for path in (REPO_ROOT, MEGATRON_ROOT):
        text = os.fspath(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_resolved_config(config_hash: str):
    path = REPO_ROOT / "runs" / config_hash / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Resolved config not found: {path}")
    return OmegaConf.load(path)


def _apply_runtime_patches(cfg) -> None:
    from src.patches import apply_patches

    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    for name in patches:
        importlib.import_module(f"src.patches.{name}")
    apply_patches(patches)


def _combined_extra_args_provider(existing_provider):
    def provider(parser):
        if existing_provider is not None:
            parser = existing_provider(parser)
        return add_slm_args(parser)

    return provider


def main() -> None:
    _prepend_paths()

    config_hash = None
    for idx, item in enumerate(sys.argv):
        if item == "--slm-config-hash" and idx + 1 < len(sys.argv):
            config_hash = sys.argv[idx + 1]
            break
    if config_hash is None:
        raise RuntimeError("--slm-config-hash must be present in torchrun args")

    cfg = _load_resolved_config(config_hash)
    _apply_runtime_patches(cfg)

    import pretrain_gpt as mg
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps

    set_startup_timestamps(
        program_start=mg._PROGRAM_START_TIME,
        main_entry=mg.time.time(),
    )
    mg.train_valid_test_datasets_provider.is_distributed = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        mg.train_valid_test_datasets_provider,
        partial(mg.model_provider, mg.gpt_builder),
        ModelType.encoder_or_decoder,
        mg.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=_combined_extra_args_provider(
            mg.add_modelopt_args if mg.has_nvidia_modelopt else None
        ),
        store=store,
        get_embedding_ranks=mg.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
