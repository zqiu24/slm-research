# Copyright (c) 2025. POET / POET-X integration for Megatron-LM.
#
# This script is a thin wrapper around ``pretrain_gpt.py`` that:
#   1. Registers POET CLI arguments with Megatron's parser.
#   2. Wraps ``model_provider`` so that, right after the GPT model is built
#      (and before DDP wrapping), every eligible tensor-parallel linear is
#      replaced with a POET-parameterized forward.
#   3. Installs a post-optimizer-step hook that runs the periodic
#      merge-then-reinitialize step.
#
# Reference:
#   - POET:   https://arxiv.org/abs/2506.08001
#   - POET-X: https://arxiv.org/abs/2603.05500
#   - Code:   https://github.com/Sphere-AI-Lab/poet
"""POET / POET-X pretraining entry point for Megatron-LM."""

import logging
import os
import sys

# Make the vendored poet_torch package importable (same convention as the
# baseline training scripts which export PYTHONPATH=<repo root>).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch

# Raise Dynamo's compile/recompile caps *before* we import any module that
# triggers @torch.compile tracing (poet_torch.core.ops.forward_core is
# decorated at import time).
#
# Why this matters: forward_core uses ``fullgraph=True``. In recent PyTorch
# that sets ``one_graph=True`` inside the compiler, which guards against
# excessive recompilation by raising ``FailOnRecompileLimitHit`` once
# ``config.recompile_limit`` (default 8) is hit *per frame*. With MoE +
# several distinct POET linear shapes (qkv / proj / fc1 / fc2 / router /
# shared-expert) and the POET merge-then-reinit bumping every weight and
# perm buffer's ``_version`` counter every ``--poet-merge-interval`` steps,
# guard failures stack up quickly and step 201 dies with
# ``recompile_limit reached``. We both (a) bump the caps here and (b)
# call ``torch._dynamo.reset()`` right after each merge in the adapter.
try:
    import torch._dynamo as _dynamo_cfg
    # ``cache_size_limit`` controls eviction; ``recompile_limit`` is the
    # one-graph hard fail threshold introduced in newer PyTorch. Keep both
    # well above the product (num wrapped linears x num merge events)
    # expected in a typical training run.
    _dynamo_cfg.config.cache_size_limit = max(
        _dynamo_cfg.config.cache_size_limit, 512
    )
    if hasattr(_dynamo_cfg.config, "recompile_limit"):
        _dynamo_cfg.config.recompile_limit = max(
            _dynamo_cfg.config.recompile_limit, 512
        )
    if hasattr(_dynamo_cfg.config, "accumulated_cache_size_limit"):
        _dynamo_cfg.config.accumulated_cache_size_limit = max(
            _dynamo_cfg.config.accumulated_cache_size_limit, 1024
        )
    if hasattr(_dynamo_cfg.config, "accumulated_recompile_limit"):
        _dynamo_cfg.config.accumulated_recompile_limit = max(
            _dynamo_cfg.config.accumulated_recompile_limit, 1024
        )
except Exception:  # pragma: no cover
    pass

import pretrain_gpt as _base  # re-uses dataset / forward-step / loss definitions
from megatron.core import poet_adapter
from megatron.core.enums import ModelType
from megatron.training import get_args, inprocess_restart, pretrain, print_rank_0

# ---------------------------------------------------------------------------
# Megatron `local` + RMSNorm workaround.
#
# With --transformer-impl local (which we require so POET can wrap Megatron's
# native ColumnParallelLinear / RowParallelLinear; TE's parallel linears don't
# subclass those), RMSNorm goes through a mixed code path:
#
#   - Per-layer `input_layernorm` / `pre_mlp_layernorm` correctly resolve to
#     `WrappedTorchNorm` via `LocalSpecProvider.layer_norm(rms_norm=True)`,
#     which mutates `megatron.core.models.backends.LNImpl`.
#   - But `get_gpt_decoder_block_spec` reads `gpt_layer_specs.LNImpl` (a
#     *separate* module-level binding imported from `fused_layer_norm`), which
#     stays `FusedLayerNorm`. That is used for `final_layernorm`.
#   - `FusedLayerNorm.__init__` asserts `config.normalization == "LayerNorm"`,
#     so RMSNorm crashes at final_layernorm construction.
#
# Both POET and POET-X YAMLs use RMSNorm, so we force `gpt_layer_specs.LNImpl`
# to `WrappedTorchNorm` here at import time. This is a no-op for LayerNorm
# configs (WrappedTorchNorm routes to torch.nn.LayerNorm then) and essential
# for RMSNorm. Perf impact is negligible: final_layernorm is a single norm
# and torch.nn.RMSNorm is a native C++ op in PyTorch >= 2.4.
# ---------------------------------------------------------------------------
from megatron.core.models.gpt import gpt_layer_specs as _gls
from megatron.core.transformer.torch_norm import WrappedTorchNorm as _WrappedTorchNorm

_gls.LNImpl = _WrappedTorchNorm

try:
    from megatron.post_training.arguments import add_modelopt_args
    _has_modelopt = True
except ImportError:
    _has_modelopt = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extra-args provider: POET args + (optionally) the existing ModelOpt args.
# ---------------------------------------------------------------------------


def _extra_args_provider(parser):
    poet_adapter.add_poet_args(parser)
    if _has_modelopt:
        add_modelopt_args(parser)
    return parser


# ---------------------------------------------------------------------------
# POET-aware model provider
# ---------------------------------------------------------------------------


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build the GPT model and, if enabled, install POET on its parallel linears."""
    model = _base.model_provider(
        pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
    )

    args = get_args()
    if not getattr(args, "use_poet", False):
        return model

    if getattr(args, "poet_wrap_moe_experts", False) and getattr(args, "moe_grouped_gemm", False):
        raise RuntimeError(
            "--poet-wrap-moe-experts only applies when routed experts are built as SequentialMLP "
            "with per-expert Megatron TP linears; disable --moe-grouped-gemm (GroupedMLP / "
            "TEGroupedMLP fused weights are not ColumnParallelLinear/RowParallelLinear)."
        )

    poet_exclude_ancestors = args.poet_exclude_ancestors
    if poet_exclude_ancestors is None and getattr(args, "poet_wrap_moe_experts", False):
        # Default install skips `.experts.` / `local_experts`; those rules exist so GroupedGEMM +
        # shared_experts coexist. Narrow exclusion to fused grouped stacks only — SequentialMLP
        # routed experts remain eligible.
        poet_exclude_ancestors = ("grouped_mlp", "te_grouped_mlp")

    # Many poetx inductor caches scale with (#wrapped layers × merges); routed experts multiply
    # layer count massively at EP=local-expert cardinality.
    if getattr(args, "poet_wrap_moe_experts", False):
        try:
            import torch._dynamo as _dyn

            hi = 2048
            _dyn.config.cache_size_limit = max(_dyn.config.cache_size_limit, hi)
            if hasattr(_dyn.config, "recompile_limit"):
                _dyn.config.recompile_limit = max(_dyn.config.recompile_limit, hi)
            if hasattr(_dyn.config, "accumulated_cache_size_limit"):
                _dyn.config.accumulated_cache_size_limit = max(
                    _dyn.config.accumulated_cache_size_limit, hi * 2
                )
            if hasattr(_dyn.config, "accumulated_recompile_limit"):
                _dyn.config.accumulated_recompile_limit = max(
                    _dyn.config.accumulated_recompile_limit, hi * 2
                )
        except Exception:  # pragma: no cover
            pass

    # Install POET *before* the model is wrapped by DDP (pretrain does the
    # wrapping after model_provider returns).
    n_wrapped = poet_adapter.install_poet_in_model(
        model,
        block_size=args.poet_block_size,
        normalize_weights=args.poet_normalize_weights,
        exclude_modules=args.poet_exclude_modules,
        exclude_ancestors=poet_exclude_ancestors,
        variant=getattr(args, "poet_variant", "poet"),
        mem_efficient=getattr(args, "poet_mem_efficient", False),
    )

    # Print a summary on rank 0.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        poet_p = sum(
            p.numel()
            for n, p in model.named_parameters()
            if n.endswith("oft_R") and p.requires_grad
        )
        print_rank_0(
            f"[POET] variant={getattr(args, 'poet_variant', 'poet')} | "
            f"wrapped {n_wrapped} parallel-linear layers | "
            f"block_size={args.poet_block_size} | merge_interval={args.poet_merge_interval} | "
            f"mem_efficient={args.poet_mem_efficient} | quantize={args.poet_quantize}"
        )
        print_rank_0(
            f"[POET] params: total={total:,} trainable={trainable:,} oft_R={poet_p:,} "
            f"(oft_R = {100.0 * poet_p / max(trainable, 1):.2f}% of trainable)"
        )

    return model


# ---------------------------------------------------------------------------
# Merge hook: patch ``train_step`` so merge_then_reinitialize runs after each
# successful optimizer.step().
# ---------------------------------------------------------------------------


def _install_merge_hook():
    """Monkey-patch ``megatron.training.training.train_step`` to call the
    POET merge routine on every optimizer step that completed successfully."""
    from megatron.training import training as _training_module

    orig_train_step = _training_module.train_step

    def train_step_with_poet_merge(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
    ):
        result = orig_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
        )
        args = get_args()
        if getattr(args, "use_poet", False) and args.poet_merge_interval > 0:
            # ``args.curr_iteration`` is the 0-indexed step of the training
            # loop; ``iteration`` (inside the pretrain loop) is 1-indexed. We
            # use curr_iteration so the merge fires on step 200, 400, ....
            step = getattr(args, "curr_iteration", 0) + 1
            # Only merge if the step actually produced a successful update.
            # ``result`` is a tuple; the 2nd element is ``skipped_iter``.
            try:
                skipped_iter = result[1]
            except Exception:
                skipped_iter = 0
            if skipped_iter == 0:
                merged = poet_adapter.merge_all_poet_layers(
                    model,
                    step=step,
                    merge_interval=args.poet_merge_interval,
                    optimizer=optimizer,
                )
                if merged:
                    print_rank_0(f"[POET] merge-then-reinitialize at step {step}")
        return result

    _training_module.train_step = train_step_with_poet_merge


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    _base.train_valid_test_datasets_provider.is_distributed = True

    _install_merge_hook()

    # Put oft_R in its own param group: weight_decay=0 and a scaled LR
    # (defaults reproduce the reference POETAdamW ratios). Gated inside the
    # hook on --use-poet so non-POET runs are unaffected.
    poet_adapter.install_poet_optimizer_hook()

    pretrain_wrapped, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain_wrapped(
        _base.train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        _base.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=_extra_args_provider,
        store=store,
    )
