"""POET linear-replacement helpers.

Ported from fork 2's ``megatron/poet_integration.py`` (commit bb43fa063).
The Megatron-specific type list (``ColumnParallelLinear`` /
``TEColumnParallelLinear`` / ...) is discovered lazily so unit tests can
pass in plain ``torch.nn.Linear`` via ``extra_linear_types``.

POET requires the model to be built with ``config.transformer_impl='local'``
so that ``TELayerNormColumnParallelLinear`` (fused norm + linear) is not
materialised — the patch in ``src/patches/poet_unfuse_te_impl.py`` enforces
that automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
from poet_torch import POETLinear

from src.optim import poet_cache as _poet_cache

logger = logging.getLogger(__name__)

# Leaf names produced by ``src.model.unfuse_linears`` when a fused linear is
# unfused. A non-divisible layer with one of these names is a hard error rather
# than a silent skip (see ``replace_linears_with_poet``).
_UNFUSED_SEGMENT_NAMES = frozenset(
    {"linear_q", "linear_k", "linear_v", "linear_fc1_gate", "linear_fc1_up"}
)

# Attention projections that take head-aligned rotation, and which side carries
# the heads. q/k/v rows are heads (out); the output projection's cols are (in).
_HEAD_ALIGNED_SIDES = {
    "linear_q": "out",
    "linear_k": "out",
    "linear_v": "out",
    "linear_proj": "in",
}


def _copy_and_init_weight(pl, child, init_type, mup_alpha):
    """Copy child's weight (+bias) into the POET layer's frozen base, applying
    init_type. Shared by the stock and head-aligned branches."""
    out_f, in_f = child.weight.shape
    has_bias = child.bias is not None and child.bias.numel() > 0
    with torch.no_grad():
        w = child.weight.data.clone()
        if init_type == "normalized":
            w = w / torch.norm(w, dim=1, keepdim=True)
        elif init_type == "mup_normalized":
            d_in = torch.tensor(float(in_f))
            d_out = torch.tensor(float(out_f))
            w = w / torch.norm(w, dim=1, keepdim=True)
            target = mup_alpha * torch.sqrt(d_out / d_in)
            current = torch.linalg.norm(w.float(), ord=2).item()
            w = w * (target / current).to(dtype=w.dtype, device=w.device)
        pl.weight.copy_(w.to(pl.weight.dtype))
        if has_bias:
            pl.bias.copy_(child.bias.data.to(pl.bias.dtype))


class POETMegatronLinear(nn.Module):
    """Wraps a :class:`POETLinear` to match Megatron's parallel-linear
    calling convention.

    ``ColumnParallelLinear`` and ``RowParallelLinear`` both return
    ``(output, output_bias)``. This wrapper preserves that convention so
    callers downstream of the swap don't notice the substitution.
    """

    def __init__(self, poet_linear: POETLinear, skip_bias_add: bool = False):
        super().__init__()
        self.poet_linear = poet_linear
        self._skip_bias_add = skip_bias_add
        # Expose weight / bias for DDP and Megatron introspection.
        self.weight = poet_linear.weight
        self.bias = poet_linear.bias

    def forward(self, input_: torch.Tensor, weight=None, **kw):
        output = self.poet_linear(input_)
        return output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Distributed-checkpoint (``torch_dist``) sharding for the POET wrapper.

        Megatron's ``torch_dist`` save walks the model and calls
        ``sharded_state_dict()`` on every submodule (transformer_block → layer →
        mlp/attention → linear). The ``ColumnParallelLinear`` /
        ``RowParallelLinear`` we replaced implement this; this plain
        ``nn.Module`` wrapper must too, or the save aborts at the first
        checkpoint with ``AttributeError: 'POETMegatronLinear' object has no
        attribute 'sharded_state_dict'``.

        POET runs at ``tensor_parallel_size == 1`` (the parallelism rules pin
        tp=1 for <3e9-param models, i.e. every POET scale), so every tensor —
        the frozen base ``weight``/``bias``, the trainable ``oft_R_*`` rotations,
        and the permutation / skew-index buffers — is *replicated*: no TP axis
        map, all emitted as fully-replicated ShardedTensors (DP replicas tracked
        by the helper).

        ``__init__`` aliases ``self.weight``/``self.bias`` to
        ``poet_linear.weight``/``.bias`` (the same objects, for DDP / Megatron
        introspection), so they appear twice in ``state_dict()``. We drop the
        bare aliases here so the frozen base weight — the bulk of the checkpoint
        — is serialized exactly once, under ``poet_linear.*``.
        """
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )

        state_dict = self.state_dict(prefix="", keep_vars=True)
        # Aliases of poet_linear.weight / poet_linear.bias (identical objects).
        state_dict.pop("weight", None)
        state_dict.pop("bias", None)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map=None,  # replicated (tp=1)
            sharded_offsets=sharded_offsets,
        )


def _megatron_linear_types() -> tuple[type, ...]:
    """Discover Megatron linear types; empty tuple if Megatron isn't importable.

    We catch ``Exception`` (not just ``ImportError``) because Megatron's
    top-level import eagerly loads ``transformer_engine``, which raises
    ``OSError: libcublas.so.12`` on CPU-only nodes. Returning an empty tuple
    means the caller falls back to ``extra_linear_types``.
    """
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
    except Exception:
        return ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        return (
            ColumnParallelLinear,
            RowParallelLinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
    except Exception:
        return (ColumnParallelLinear, RowParallelLinear)


def _fused_layernorm_linear_types() -> tuple[type, ...]:
    """Modules POET must refuse to replace (the unfused-spec error case)."""
    out: tuple[type, ...] = ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
        )

        out += (TELayerNormColumnParallelLinear,)
    except Exception:
        pass
    try:
        from megatron.core.tensor_parallel.inference_layers import (
            InferenceLayerNormColumnParallelLinear,
        )

        out += (InferenceLayerNormColumnParallelLinear,)
    except Exception:
        pass
    return out


def replace_linears_with_poet(
    model: nn.Module,
    *,
    block_size: int = 256,
    block_count: int | None = None,
    init_type: str = "normalized",
    mup_alpha: float = 1.0,
    skip_lm_head: bool = True,
    extra_linear_types: Iterable[type] = (),
    cache_mode: str = "none",
    parameterization: str = "cayley",
    freeze_output_rotation: bool = False,
    head_aligned_attn: bool = False,
    head_dim: int | None = None,
    head_resid_block_count: int = 1,
    resid_permute: bool = True,
    single_step_fast: bool = False,
    single_step_native: bool = False,
    single_step_x: bool = False,
    single_step_x_alternating: bool = False,
    lie_alternating: bool = False,
    alternate_every: int = 1,
) -> int:
    """Walk ``model`` and replace each parallel-linear with a
    :class:`POETMegatronLinear`.

    Returns the number of replacements.

    Raises ``RuntimeError`` if the model still has fused LayerNormLinear
    modules — those carry a layer-norm payload that POET would silently
    drop. The caller must rebuild the model with
    ``config.transformer_impl == 'local'`` first; the patch in
    ``src/patches/poet_unfuse_te_impl.py`` does that automatically.
    """
    fused = _fused_layernorm_linear_types()
    linear_types: tuple[type, ...] = _megatron_linear_types() + tuple(extra_linear_types)
    if not linear_types:
        raise RuntimeError(
            "No replaceable linear types found. Pass "
            "extra_linear_types=(nn.Linear,) for tests, or make sure "
            "megatron is importable."
        )

    if parameterization == "exp" and cache_mode != "none":
        raise ValueError(
            "parameterization='exp' is not supported with cache_mode != 'none' "
            "(the cached Cayley path is a documented dead-end; use cache_mode='none')."
        )

    replaced = 0
    skipped = 0

    def _walk(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced, skipped
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            if fused and isinstance(child, fused):
                raise RuntimeError(
                    f"[POET] Fused LayerNormLinear at {full} "
                    f"({type(child).__name__}). Rebuild with "
                    "config.transformer_impl='local' before applying POET."
                )

            if isinstance(child, linear_types):
                if skip_lm_head and "output_layer" in full:
                    skipped += 1
                    continue
                if head_aligned_attn and name == "linear_qkv":
                    raise ValueError(
                        f"[POET] head_aligned_attn requires unfused q/k/v "
                        f"(set base.model.unfuse_qkv=true); found fused {full}"
                    )
                if head_aligned_attn and name in _HEAD_ALIGNED_SIDES:
                    if head_dim is None:
                        raise ValueError("[POET] head_aligned_attn requires head_dim")
                    out_f, in_f = child.weight.shape
                    has_bias = child.bias is not None and child.bias.numel() > 0
                    head_side = _HEAD_ALIGNED_SIDES[name]
                    if single_step_x:
                        # POETX-native head-aligned: forward-frame, identity perm on the
                        # head side + a real permuted multi-block residual side.
                        from poet_torch import HeadAlignedPOETXLinear

                        pl = HeadAlignedPOETXLinear(
                            in_features=in_f,
                            out_features=out_f,
                            head_side=head_side,
                            head_dim=head_dim,
                            head_resid_block_count=head_resid_block_count,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternating=(single_step_x and lie_alternating),
                            alternate_every=alternate_every,
                        )
                        _copy_and_init_weight(pl, child, init_type, mup_alpha)
                        pl.bake_perms_into_weight()  # POETX stores the forward frame
                    else:
                        from poet_torch import HeadAlignedPOETLinear

                        resid_kwargs = (
                            {"resid_block_count": block_count}
                            if block_count is not None
                            else {"resid_block_size": block_size}
                        )
                        pl = HeadAlignedPOETLinear(
                            in_features=in_f,
                            out_features=out_f,
                            head_side=head_side,
                            head_dim=head_dim,
                            resid_permute=resid_permute,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            **resid_kwargs,
                        )
                        _copy_and_init_weight(pl, child, init_type, mup_alpha)
                    pl.single_step_fast = single_step_fast or single_step_native or single_step_x
                    wrapper = POETMegatronLinear(
                        pl, skip_bias_add=getattr(child, "skip_bias_add", False)
                    )
                    setattr(parent, name, wrapper)
                    replaced += 1
                    continue
                out_f, in_f = child.weight.shape
                # block_count (when set) takes precedence over block_size.
                divisor = block_count if block_count is not None else block_size
                if in_f % divisor != 0 or out_f % divisor != 0:
                    # An unfused sub-projection (from src.model.unfuse_linears)
                    # that POET can't wrap is a hard error: the user asked for it
                    # to be POET-ised, so fail fast rather than silently skip.
                    if name in _UNFUSED_SEGMENT_NAMES:
                        label = "block_count" if block_count is not None else "block_size"
                        raise ValueError(
                            f"[POET] unfused segment {full} dims (in={in_f}, out={out_f}) "
                            f"not divisible by {label}={divisor}. Pick a compatible "
                            f"block_size/block_count, or disable unfusing this layer."
                        )
                    logger.info(
                        "[POET] skip %s: dims (%d, %d) not divisible by %s=%d",
                        full,
                        in_f,
                        out_f,
                        "block_count" if block_count is not None else "block_size",
                        divisor,
                    )
                    skipped += 1
                    continue

                # Exactly one of bsz / block_count is forwarded to POETLinear.
                if block_count is not None:
                    block_kwargs = {"block_count": block_count}
                else:
                    block_kwargs = {"bsz": block_size}

                has_bias = child.bias is not None and child.bias.numel() > 0
                if cache_mode == "none":
                    if single_step_x and single_step_x_alternating:
                        from poet_torch import AlternatingPOETXLinear as _PoetCls

                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    elif single_step_x:
                        # Integrated path: a plain POETXLinear that carries the
                        # alternating flag (both-momenta forward/backward; the merge
                        # driver folds only the active side). lie_alternating=False
                        # builds the ordinary both-sides POETXLinear.
                        from poet_torch import POETXLinear as _PoetCls

                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternating=lie_alternating,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    else:
                        if single_step_native:
                            from poet_torch import SingleStepPOETLinear as _PoetCls
                        else:
                            _PoetCls = POETLinear  # noqa: N806
                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            **block_kwargs,
                        )
                else:
                    pl = _poet_cache.CachedPOETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                        **block_kwargs,
                    )
                    _poet_cache.register_poet_layer(pl)
                if freeze_output_rotation and hasattr(pl, "oft_R_out"):
                    # Single-sided POET: keep R_out = identity (oft_R_out inits to
                    # zero) and never train it. requires_grad=False is set here,
                    # pre-DDP, so oft_R_out is excluded from the grad buffer and the
                    # optimizer param groups (which only take requires_grad params).
                    pl.oft_R_out.requires_grad_(False)
                _copy_and_init_weight(pl, child, init_type, mup_alpha)
                pl.single_step_fast = single_step_fast
                if single_step_x:
                    # POETX stores the forward-frame weight; convert the just-copied
                    # natural weight Wx = W[perm_out][:,perm_in] (one-time, at build).
                    pl.bake_perms_into_weight()

                wrapper = POETMegatronLinear(
                    pl, skip_bias_add=getattr(child, "skip_bias_add", False)
                )
                setattr(parent, name, wrapper)
                replaced += 1
            else:
                _walk(child, full)

    _walk(model)
    logger.info("[POET] replaced %d, skipped %d", replaced, skipped)
    return replaced
