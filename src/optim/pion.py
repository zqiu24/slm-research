"""Builder for the ``pion`` optimizer (Megatron integration).

Wraps the vendored single-process Pion algorithm (``src/optim/_pion.py``) in
Megatron's optimizer machinery as a ``ChainedOptimizer``: Pion drives the 2-D
matrix (non-embedding) weights and a stock Megatron AdamW drives everything else
(embeddings, norms, biases, LM head). Single-GPU dev scope only.

Mirrors the upstream ``get_megatron_pion_optimizer``
(third_party/pion/megatron-lm/megatron/core/optimizer/pion.py) but resolves the
Megatron optimizer primitives lazily from ``megatron.core.optimizer`` — the
UN-patched originals — so the chained-Adam call does NOT recurse back into the
``pion_optimizer_setup`` patch (which only rebinds the names in
``megatron.training.training``). Same no-recursion design as ``src/optim/poet.py``.
Reached via ``src/patches/pion_optimizer_setup.py``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

# Lazy handles; populated by _resolve_megatron_handles on first build. Kept as
# module globals so unit tests can monkeypatch them without importing Megatron.
_get_param_groups = None
get_megatron_optimizer = None
ChainedOptimizer = None
Float16OptimizerWithFloat16Params = None
FP32Optimizer = None


def _resolve_megatron_handles() -> None:
    """Import Megatron optimizer primitives on first use.

    Resolved from ``megatron.core.optimizer`` (the originals), NOT from
    ``megatron.training.training`` (which the pion_optimizer_setup patch wraps) —
    so the chained-Adam build below does not recurse into the patch.
    """
    global _get_param_groups, get_megatron_optimizer, ChainedOptimizer
    global Float16OptimizerWithFloat16Params, FP32Optimizer
    if _get_param_groups is not None:
        return
    from megatron.core.optimizer import _get_param_groups as _gpg
    from megatron.core.optimizer import get_megatron_optimizer as _gmo
    from megatron.core.optimizer.optimizer import ChainedOptimizer as _ChainedOptimizer
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params as _Float16OptimizerWithFloat16Params,
    )
    from megatron.core.optimizer.optimizer import FP32Optimizer as _FP32Optimizer

    _get_param_groups = _gpg
    get_megatron_optimizer = _gmo
    ChainedOptimizer = _ChainedOptimizer
    Float16OptimizerWithFloat16Params = _Float16OptimizerWithFloat16Params
    FP32Optimizer = _FP32Optimizer


def _matrix_param_groups(
    model_chunks: list[Any],
    config: Any,
    config_overrides: dict | None,
    matrix_params: list[Any],
) -> list[dict]:
    """Restrict Megatron's standard param groups to the Pion matrix params."""
    matrix_param_ids = {id(p) for p in matrix_params}
    groups: list[dict] = []
    for group in _get_param_groups(model_chunks, config, config_overrides):
        params = [p for p in group["params"] if id(p) in matrix_param_ids]
        if params:
            new_group = dict(group)
            new_group["params"] = params
            groups.append(new_group)
    return groups


def get_megatron_pion_optimizer(
    config: Any,
    model_chunks: list[Any],
    *,
    config_overrides: dict | None = None,
    use_gloo_process_groups: bool = True,
) -> Any:
    from megatron.core import parallel_state as mpu

    from src.optim._pion import PionOptimizer

    if config.use_distributed_optimizer:
        raise ValueError("pion does not support the distributed optimizer (single-GPU dev only).")
    if config.fp16:
        raise ValueError("pion does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("pion does not support tensor parallelism > 1 (single-GPU dev only).")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("pion does not support pipeline parallelism > 1 (single-GPU dev only).")

    _resolve_megatron_handles()

    # The Pion experiment routes through --optimizer adam (the stock path builds
    # an AdamOptimizerConfig). Keep config.optimizer == "adam" so the chained-Adam
    # build below takes the standard path.
    config.optimizer = "adam"

    matrix_params: list[Any] = []
    non_matrix_params: list[Any] = []
    qkv_split_shapes: tuple[int, int, int] | None = None
    split_fc1_up_gate = False

    for model_chunk in model_chunks:
        num_attention_heads = getattr(model_chunk.config, "num_attention_heads", None)
        num_query_groups = getattr(model_chunk.config, "num_query_groups", None)
        kv_channels = getattr(model_chunk.config, "kv_channels", None)
        if (
            num_attention_heads is not None
            and num_query_groups is not None
            and kv_channels is not None
        ):
            qkv_split_shapes = (
                num_attention_heads // num_query_groups * kv_channels,
                kv_channels,
                kv_channels,
            )
        gated_linear_unit = getattr(model_chunk.config, "gated_linear_unit", False)
        split_fc1_up_gate = gated_linear_unit and getattr(config, "pion_split_gate", True)

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            ):
                param._pion_param_name = name
                if "linear_qkv.weight" in name:
                    param.is_qkv = True
                if "linear_fc1.weight" in name and split_fc1_up_gate:
                    param.is_fc1_up_gate = True
                matrix_params.append(param)
            else:
                non_matrix_params.append(param)

    # Diagnostic: surface the routing split (mirror muon_kimi.py:101-109).
    logger.info(
        "pion: %d matrix params (2D non-embedding), %d adamw params",
        len(matrix_params),
        len(non_matrix_params),
    )
    if not matrix_params:
        logger.warning("pion: no 2D non-embedding params found — Pion is a no-op (pure AdamW).")

    lr = float(config.lr if config.lr is not None else 1e-4)
    matrix_param_groups = _matrix_param_groups(
        model_chunks, config, config_overrides, matrix_params
    )
    if not matrix_param_groups:
        matrix_param_groups = [
            {
                "params": matrix_params,
                "max_lr": lr,
                "min_lr": config.min_lr,
                "wd_mult": 1.0,
                "lr_mult": 1.0,
                "is_expert_parallel": False,
                "default_config": True,
            }
        ]

    degree = getattr(config, "pion_degree", 2)
    pion_scaling = getattr(config, "pion_scaling", "rms")
    pion_rms = getattr(config, "pion_rms", 0.2)
    pion_momentum = getattr(config, "pion_momentum", "none")
    pion_use_second_momentum = getattr(config, "pion_use_second_momentum", None)
    pion_update_side = getattr(config, "pion_update_side", "both")
    pion_qkv_split_granularity = getattr(config, "pion_qkv_split_granularity", None)
    if pion_qkv_split_granularity is None:
        pion_qkv_split_granularity = (
            "head" if getattr(config, "pion_split_qkv_per_head", True) else "qkv"
        )
    pion_exp_map = getattr(config, "pion_exp_map", "exp_truncated")
    adam_eps = getattr(config, "adam_eps", 1e-8)
    pion_beta1 = getattr(config, "pion_beta1", 0.9)
    pion_beta2 = getattr(config, "pion_beta2", 0.999)

    for group in matrix_param_groups:
        group["degree"] = degree
        group["pion_scaling"] = pion_scaling
        group["pion_rms"] = pion_rms
        group["pion_momentum"] = pion_momentum
        group["pion_use_second_momentum"] = pion_use_second_momentum
        group["pion_update_side"] = pion_update_side
        group["pion_qkv_split_granularity"] = pion_qkv_split_granularity
        # The next five knobs are intentionally NOT plumbed through the slm-research
        # CLI/patch: the legacy momentum-blend selectors (12/first/second_momentum)
        # and the per-update CSV diagnostics are out of scope for this dev port. They
        # always resolve to their upstream defaults ("none"/None/1); the group dict is
        # still populated so the vendored PionOptimizer reads a complete config.
        group["pion_12_momentum"] = getattr(config, "pion_12_momentum", "none")
        group["pion_first_momentum"] = getattr(config, "pion_first_momentum", "none")
        group["pion_second_momentum"] = getattr(config, "pion_second_momentum", "none")
        group["pion_exp_map"] = pion_exp_map
        group["pion_update_csv"] = getattr(config, "pion_update_csv", None)
        group["pion_update_csv_interval"] = getattr(config, "pion_update_csv_interval", 1)
        group["adam_eps"] = adam_eps
        group["pion_beta1"] = pion_beta1
        group["pion_beta2"] = pion_beta2

    pion_optimizer = PionOptimizer(
        matrix_param_groups,
        lr=lr,
        betas=(pion_beta1, pion_beta2),
        weight_decay=config.weight_decay,
        degree=degree,
        split_qkv=getattr(config, "pion_split_qkv", True),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        split_fc1_up_gate=split_fc1_up_gate,
        is_fc1_up_gate_fn=lambda p: getattr(p, "is_fc1_up_gate", False),
        split_qkv_per_head=getattr(config, "pion_split_qkv_per_head", True),
        qkv_split_granularity=pion_qkv_split_granularity,
        pion_scaling=pion_scaling,
        pion_rms=pion_rms,
        pion_momentum=pion_momentum,
        pion_update_side=pion_update_side,
        pion_beta1=pion_beta1,
        pion_beta2=pion_beta2,
    )

    def pion_init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["step"] = 0

    if config.bf16:
        wrapped = Float16OptimizerWithFloat16Params(
            pion_optimizer, config, None, pion_init_state_fn
        )
    else:
        wrapped = FP32Optimizer(pion_optimizer, config, pion_init_state_fn)

    optimizers: list[Any] = [wrapped]

    # Build the stock Megatron AdamW for the NON-matrix params. Freeze the matrix
    # params first so _get_param_groups (which skips requires_grad=False) hands the
    # standard Adam path only the embeddings/norms/biases/head; unfreeze after.
    for p in matrix_params:
        p.requires_grad = False
    chained_adam = cast(
        Any,
        get_megatron_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        ),
    )
    for p in matrix_params:
        p.requires_grad = True

    optimizers += chained_adam.chained_optimizers
    wrapped.grad_stats_parallel_group = mpu.get_model_parallel_group()
    wrapped.tp_group = mpu.get_tensor_model_parallel_group()
    return ChainedOptimizer(optimizers)


__all__ = ["get_megatron_pion_optimizer"]
