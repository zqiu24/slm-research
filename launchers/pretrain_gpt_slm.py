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
    group.add_argument("--slm-config-path", type=str, required=True)
    group.add_argument(
        "--slm-optimizer",
        choices=["adamw", "muon", "poet", "ngpt_adamw", "muon_kimi"],
        default="adamw",
    )
    group.add_argument("--poet", action="store_true")
    # block_size and block_count are mutually exclusive: block_size uses one
    # shared block size on both sides; block_count gives each side `n` blocks
    # with potentially different block sizes (in/n, out/n). Defaults don't count
    # as "provided", so block_count=None + block_size=256 is the legacy default.
    poet_block_group = group.add_mutually_exclusive_group()
    poet_block_group.add_argument("--poet-block-size", type=int, default=256)
    poet_block_group.add_argument("--poet-block-count", type=int, default=None)
    group.add_argument(
        "--poet-init-type",
        choices=["none", "normalized", "mup_normalized"],
        default="normalized",
    )
    group.add_argument("--poet-mup-alpha", type=float, default=1.0)
    group.add_argument("--poet-merge-period", type=int, default=0)
    # Cadence (optimizer steps) at which the block permutation is resampled AND
    # Adam momentum is reset. 0 = fall back to --poet-merge-period (legacy: fold,
    # resample, reset all fire together). poet0 sets merge_period=1 (fold each
    # step) + reinit_period=400 so Ψ and momentum stay coherent for 400-step
    # stretches while Q is folded into W every step.
    group.add_argument("--poet-reinit-period", type=int, default=0)
    group.add_argument("--poet-scale", type=float, default=1.0)
    # Optimizer impl: default (flag absent) uses the stock Megatron-Adam path
    # (oft_R LR override + poet_merge_step momentum reset). Pass this flag to use
    # the custom POETAdam + ChainedOptimizer path instead.
    group.add_argument("--poet-use-poet-adam", action="store_true")
    # Freeze the output-side rotation (oft_R_out stays at its zero init = identity);
    # only the input-side rotation oft_R_in is trained. Single-sided POET ablation.
    group.add_argument("--poet-freeze-output-rotation", action="store_true")
    group.add_argument(
        "--poet-cache-mode",
        choices=["none", "cached_fwd", "cached_fwd_bwd"],
        default="none",
    )
    group.add_argument(
        "--poet-parameterization",
        choices=["cayley", "exp"],
        default="cayley",
    )
    group.add_argument(
        "--poet-q-optimizer", choices=["adam", "muon", "lie_algebra"], default="adam"
    )
    group.add_argument("--poet-muon-theta", type=float, default=0.1)
    group.add_argument("--poet-muon-ns-steps", type=int, default=5)
    group.add_argument("--poet-muon-momentum", type=float, default=0.95)
    # Lie-algebra momentum (q_optimizer=lie_algebra): Pion first/second-moment
    # momentum on oft_R, accumulated in the Lie algebra (persists across merges).
    group.add_argument("--poet-lie-b1", type=float, default=0.9)
    group.add_argument("--poet-lie-b2", type=float, default=0.95)
    group.add_argument("--poet-lie-eps", type=float, default=1e-8)
    group.add_argument("--poet-lie-v-mode", choices=["scalar", "elementwise"], default="scalar")
    # Architectural unfusing of fused linears (optimizer-agnostic). Applied by
    # the ``model_unfuse_linears`` patch at model-build time.
    group.add_argument("--unfuse-qkv", action="store_true")
    group.add_argument("--unfuse-fc1", action="store_true")
    group.add_argument("--ngpt", action="store_true")
    group.add_argument(
        "--ngpt-base-scale", type=float, default=None, help="1/sqrt(hidden_size) by default"
    )
    group.add_argument("--ngpt-alpha-init", type=float, default=0.05)
    group.add_argument("--ngpt-sqk-init", type=float, default=1.0)
    group.add_argument("--ngpt-suv-init", type=float, default=1.0)
    group.add_argument("--ngpt-sz-init", type=float, default=1.0)
    group.add_argument(
        "--ngpt-no-warmup",
        action="store_true",
        help="Force LR warmup steps to 0 (matches reference train.py:114)",
    )
    return parser


def _prepend_paths() -> None:
    for path in (REPO_ROOT, MEGATRON_ROOT):
        text = os.fspath(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_resolved_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Resolved config not found: {path}")
    return OmegaConf.load(path)


# Patches applied to EVERY Megatron run regardless of experiment. The logging
# ones are no-ops on the model; the diagnostic ones (overfit_single_batch,
# poet_grad_conditioning, grad_conditioning) self-disable unless their SLM_* env
# var is set, so they are inert on normal runs and stay out of the experiment
# patch_set_hash.
_ALWAYS_ON_PATCHES = (
    "wandb_trainable_params",
    "overfit_single_batch",
    "poet_grad_conditioning",
    "grad_conditioning",
)


def _resolve_runtime_patch_names(cfg) -> list[str]:
    """Experiment patches plus the always-on patches, de-duplicated, order-stable."""
    names = list(cfg.get("experiment", {}).get("patches", []) or [])
    for name in _ALWAYS_ON_PATCHES:
        if name not in names:
            names.append(name)
    return names


def _apply_runtime_patches(cfg) -> None:
    from src.patches import apply_patches

    names = _resolve_runtime_patch_names(cfg)
    for name in names:
        importlib.import_module(f"src.patches.{name}")
    apply_patches(names)


def _combined_extra_args_provider(existing_provider):
    def provider(parser):
        if existing_provider is not None:
            parser = existing_provider(parser)
        return add_slm_args(parser)

    return provider


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

    # Honor the cluster's wandb_offline flag: wandb reads WANDB_MODE natively,
    # and Megatron's wandb.init does not pass a mode. Set it before pretrain()
    # (which initializes wandb deep inside). setdefault so an explicit env
    # override (e.g. for local debugging) still wins.
    if bool(cfg.get("cluster", {}).get("wandb_offline", False)):
        os.environ.setdefault("WANDB_MODE", "offline")

    _apply_runtime_patches(cfg)

    import pretrain_gpt as mg
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps

    # Optionally re-initialize the built model to torchtitan's native llama3 init
    # scheme so the Megatron backend reproduces a torchtitan training curve. Wraps
    # the (possibly unfuse-wrapped) model_provider, so the re-init runs AFTER unfuse
    # and BEFORE DDP/optimizer setup. Gated on the resolved config, not the
    # experiment, so only configs that opt in (e.g. base/scale/300m) are affected.
    if bool(cfg.get("base", {}).get("model", {}).get("titan_init", False)):
        from src.model.titan_init import wrap_model_provider

        mg.model_provider = wrap_model_provider(mg.model_provider)

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
