"""POET / POET-X adapter for Megatron-LM.

Adapts the POET reparameterization (``W_eff = R_out @ W_0 @ R_in`` with random
block-level permutations and Cayley-Neumann parameterized orthogonal matrices)
to both Megatron's native parallel linears and Transformer-Engine's parallel
linears.

Supported linear classes:

* Megatron native: ``ColumnParallelLinear``, ``RowParallelLinear``.
  Two forward variants are available:
    - ``poet``  : materialize ``W_eff`` per forward (weight-space).
    - ``poetx`` : input-centric activation-space path (no ``W_eff``; cheaper
                  memory profile). Requires no LN fusion upstream of the
                  linear, which holds for Megatron native linears.
* Transformer-Engine: ``TELinear`` / ``TEColumnParallelLinear`` /
  ``TERowParallelLinear`` / ``TELayerNormColumnParallelLinear``. Only the
  ``poet`` (weight-space) variant is supported because
  ``TELayerNormColumnParallelLinear`` fuses RMSNorm/LN into the linear and
  RMSNorm's per-channel ``gamma`` does not commute with POET's input
  rotation. The TE path rebinds ``self.weight`` to ``W_eff`` for the
  duration of the original forward so TE's fused LN + GEMM + TP-comm path
  stays intact.

Design notes:

* Each TP rank owns a local shard of ``W`` of shape ``(out_local, in_local)``.
  POET's block-diagonal factors are applied on those local dims. Random
  permutations differ across ranks, which is fine: each rank's POET
  reparameterization is self-contained and neither alters nor depends on
  the cross-rank sharded layout.
* The base weight is frozen (``requires_grad=False``) before DDP wrapping so
  only the Cayley-parameterized ``oft_R`` tensors enter the optimizer / DDP
  grad buckets.
* Cayley-Neumann uses ``torch.ops.poet.cayley`` (Triton) when CUDA is
  available, with a pure-PyTorch fallback for CPU tests. See
  :mod:`poet_torch` at the repo root for the underlying math/kernels.
* MoE grouped-GEMM experts, embeddings, the LM head and any layer whose
  local dims are not divisible by ``block_size`` are skipped.
"""

from __future__ import annotations

import logging
import math
import types
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State object attached to each wrapped parallel linear
# ---------------------------------------------------------------------------


class PoetParallelLinearState(nn.Module):
    """Per-layer POET state for a Megatron tensor-parallel linear.

    Holds the Cayley-parameterized skew-symmetric vector ``oft_R`` (trainable)
    and the permutation buffers ``perm_in/out`` used to realize the
    block-diagonal orthogonal factors on the *local* (TP-sharded) dimensions.

    Args:
        in_features_local: input dim on this TP rank (for Row-parallel this is
            the TP-sharded size, for Column-parallel this is the full size).
        out_features_local: output dim on this TP rank (for Column-parallel
            this is the TP-sharded size, for Row-parallel this is the full
            size).
        block_size: POET block size. Both ``in_features_local`` and
            ``out_features_local`` must be divisible by this value.
        dtype: dtype of ``oft_R`` (usually bf16/fp32).
        device: device to place parameters / buffers on.
    """

    def __init__(
        self,
        in_features_local: int,
        out_features_local: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert in_features_local % block_size == 0
        assert out_features_local % block_size == 0

        self.in_features_local = in_features_local
        self.out_features_local = out_features_local
        self.block_size = block_size
        self.num_in_blocks = in_features_local // block_size
        self.num_out_blocks = out_features_local // block_size
        # Read by ``_poetx_core`` to select the mem-efficient Triton kernel
        # (``chain_layer_x_checkpoint_mem_o2``) over the default chain.
        self.mem_efficient_mode: bool = False
        n_elements = block_size * (block_size - 1) // 2

        # Trainable Cayley-parameterized skew-symmetric vectors.
        # Shape follows the upstream POET convention (out blocks + in blocks).
        self.oft_R = nn.Parameter(
            torch.zeros(
                (self.num_out_blocks + self.num_in_blocks, n_elements),
                dtype=dtype,
                device=device,
            )
        )
        # Tell Megatron's DDP/optimizer this param is not TP-sharded.  The
        # ``allreduce`` flag is filled in by ``_inherit_poet_parallel_attrs``
        # after attach, because routed MoE experts must use expert-DP groups.
        setattr(self.oft_R, "sequence_parallel", False)
        setattr(self.oft_R, "allreduce", True)
        setattr(self.oft_R, "tensor_model_parallel", False)

        rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32), persistent=False)
        self.register_buffer("cols", cols.to(torch.int32), persistent=False)

        perm_in = torch.randperm(in_features_local, device=device).to(torch.int32)
        perm_out = torch.randperm(out_features_local, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    # ---- core math ----------------------------------------------------------
    @torch.no_grad()
    def _reinit_permutations(self) -> None:
        device = self.perm_in.device
        perm_in = torch.randperm(self.in_features_local, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features_local, device=device).to(torch.int32)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in).to(torch.int32))
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out).to(torch.int32))


# ---------------------------------------------------------------------------
# Core POET math (uses the vendored poet_torch package for primitives)
# ---------------------------------------------------------------------------


def _compute_R_out_R_in(
    state: PoetParallelLinearState, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cayley-Neumann transform of ``oft_R`` -> block-diagonal orthogonals.

    Uses the Triton-backed ``torch.ops.poet.cayley`` on CUDA when available,
    otherwise falls back to the pure-PyTorch implementation so that CPU
    smoke-tests still work.
    """
    from poet_torch.core.ops import _cayley_transform_pytorch, pytorch_skew_symmetric

    Q = pytorch_skew_symmetric(
        state.oft_R.to(dtype), state.block_size, state.rows, state.cols
    )
    use_triton = (
        state.oft_R.is_cuda
        and hasattr(torch.ops, "poet")
        and hasattr(torch.ops.poet, "cayley")
    )
    if use_triton:
        R_cat = torch.ops.poet.cayley(Q)[0]
    else:
        R_cat = _cayley_transform_pytorch(Q)
    R_out, R_in = R_cat.split([state.num_out_blocks, state.num_in_blocks], dim=0)
    return R_out, R_in


def _compute_effective_weight(
    base_weight: torch.Tensor, state: PoetParallelLinearState
) -> torch.Tensor:
    """Compute ``W_eff = P_out * R_out * W_0 * R_in * P_in`` differentiably.

    Used by the **POET** variant ("weight-space materialization").
    The computation follows :class:`poet_torch.layers.POETLinear.get_effective_weight`
    but is replicated here so that it works on the arbitrary (out_local, in_local)
    shard shape of a Megatron parallel linear.
    """
    from poet_torch.core.ops import block_diag_lr_matmul

    R_out, R_in = _compute_R_out_R_in(state, base_weight.dtype)

    # base_weight is (out_local, in_local); we operate on W^T as the upstream
    # code does so that R_in multiplies on the input dim.
    tmp = base_weight.t()
    tmp = block_diag_lr_matmul(R_in, tmp, R_out)
    tmp = tmp.index_select(0, state.perm_in.long())
    tmp = tmp.index_select(1, state.perm_out.long())
    return tmp.t().contiguous()


# ---------------------------------------------------------------------------
# POET-X activation-space fast path
#
# We delegate the full permute + R_in + linear + R_out + permute chain to
# ``poet_torch.core.ops.forward_core`` from https://github.com/Sphere-AI-Lab/poet,
# which is:
#   * ``@torch.compile(fullgraph=True)`` -> fuses the whole chain into one graph.
#   * Uses ``torch.ops.poet.cayley`` (Triton) for the Cayley-Neumann transform,
#     with a fused forward + backward kernel.
#   * Switches to the ``chain_layer_x_checkpoint_mem_o2`` Triton kernel when
#     ``mem_efficient_mode=True``: activations for the chain are recomputed
#     in backward, cutting peak activation memory roughly in half.
#
# This adapter only handles the Megatron-side TP comm wrapping around the
# call; the math + numerics all live in ``poet_torch``.
# ---------------------------------------------------------------------------


def _poetx_core(
    x: torch.Tensor, state: "PoetParallelLinearState", weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the POET-X fast core (``perm + R_in + linear + R_out + perm``).

    Uses ``poet_torch.core.ops.forward_core`` so we inherit its torch.compile
    fusion and Triton Cayley kernel. ``state.mem_efficient_mode`` selects the
    additionally fused ``chain_layer_x_checkpoint_mem_o2`` Triton kernel.

    Shape contract:
        x:      (..., in_local)
        weight: (out_local, in_local)   -- frozen base weight W_0
        returns (..., out_local)
    """
    from poet_torch.core.ops import forward_core

    return forward_core(
        x,
        state.oft_R,
        state.block_size,
        state.rows,
        state.cols,
        state.perm_in,
        state.perm_in_inv,
        state.perm_out,
        state.perm_out_inv,
        state.num_in_blocks,
        state.num_out_blocks,
        weight,
        bias,
        getattr(state, "mem_efficient_mode", False),
    )


# ---------------------------------------------------------------------------
# TE (transformer_engine) weight-space forward
#
# TE parallel linears don't subclass Megatron's native Column/RowParallelLinear,
# their forward signatures don't accept a ``weight=`` kwarg, and
# ``TELayerNormColumnParallelLinear`` fuses RMSNorm/LN into the linear --
# so the activation-space POET-X path (which requires inserting R_in between
# LN and the GEMM) is not applicable without breaking the fusion. Instead we
# materialize ``W_eff`` per forward and temporarily rebind ``self.weight`` so
# TE's fused LN+GEMM+TP-comm path stays intact. Autograd propagates
# ``grad_W_eff`` back into ``oft_R`` through ``_compute_effective_weight``.
# ---------------------------------------------------------------------------


def _te_poet_weight_swap_forward(self, *args, **kwargs):
    """POET weight-space forward for Transformer-Engine parallel linears.

    Works uniformly for ``TELinear`` / ``TEColumnParallelLinear`` /
    ``TERowParallelLinear`` / ``TELayerNormColumnParallelLinear`` because the
    only thing we need to touch is ``self.weight``.

    Implementation note: we keep the original Parameter registered in
    ``self._parameters["weight"]`` and just shadow attribute lookup by placing
    ``W_eff`` into ``self.__dict__["weight"]``. Python resolves ``self.weight``
    via ``__dict__`` before falling back to ``nn.Module.__getattr__`` / the
    ``_parameters`` table, so the forward sees ``W_eff`` while TE internals
    that call ``self.named_parameters()`` still report ``weight`` as the first
    direct parameter (TE's ``linear.py`` does
    ``getattr(self, list(self.named_parameters())[0][0]).device``; if we pop
    ``weight`` the first entry becomes the dotted path
    ``"_poet_state.oft_R"`` and that ``getattr`` raises). The transient
    shadow is always cleared in ``finally`` so grads / DDP / checkpointing
    still see ``weight`` as a real Parameter.
    """
    state = self._poet_state
    W_eff = _compute_effective_weight(self.weight, state)

    self.__dict__["weight"] = W_eff
    try:
        return self._poet_orig_forward(*args, **kwargs)
    finally:
        self.__dict__.pop("weight", None)


# ---------------------------------------------------------------------------
# Monkey-patched forwards for Column / Row parallel linears
# ---------------------------------------------------------------------------


def _column_parallel_poet_forward(self, input_, weight=None, runtime_gather_output=None):
    """POET-aware forward for ``ColumnParallelLinear``.

    Replaces ``self.weight`` with the reparameterized ``W_eff`` in the forward
    pass. Falls back to the original forward if an external ``weight`` is
    supplied (e.g. tied embeddings).
    """
    if weight is None and self.weight is not None and hasattr(self, "_poet_state"):
        weight = _compute_effective_weight(self.weight, self._poet_state)
    return self._poet_orig_forward(input_, weight=weight, runtime_gather_output=runtime_gather_output)


def _row_parallel_poet_forward(self, input_):
    """POET-aware forward for ``RowParallelLinear``.

    We cannot use the ``weight=`` kwarg trick because ``RowParallelLinear.forward``
    does not expose it; so we replicate the minimal forward body, substituting
    ``W_eff`` in place of ``self.weight``.
    """
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )

    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

    weight = _compute_effective_weight(self.weight, self._poet_state)

    allreduce_dgrad = False
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=False,
        tp_group=None,
        grad_output_buffer=None,
    )

    if self.explicit_expert_comm:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


# ---------------------------------------------------------------------------
# POET-X (input-centric) forwards. These avoid materializing W_eff: R_in is
# applied to activations before the linear, R_out after. Memory drops from
# O(out * in) per forward (W_eff) to O(tokens * features) for the rotated
# activations, which is typically much smaller.
# ---------------------------------------------------------------------------


def _column_parallel_poetx_forward(self, input_, weight=None, runtime_gather_output=None):
    """POET-X fast forward for ``ColumnParallelLinear``.

    The full ``perm -> R_in -> linear -> R_out -> perm`` chain is run by
    :func:`_poetx_core` (torch.compile + Triton). TP comm wrapping stays here
    so that this path is correct for TP=1 today and forward-compatible for
    TP>1 if the caller ever enables it (note: the per-rank random
    permutations make cross-rank R_out on ``RowParallelLinear`` ill-defined
    at TP>1 -- see :func:`_row_parallel_poetx_forward`).

    Layout (TP=1):
        y = forward_core(x, oft_R, ..., W_0, bias)
    """
    if weight is not None:
        # External weight override (e.g., tied embeddings). POET-X cannot be
        # applied here; fall back to the original forward.
        return self._poet_orig_forward(
            input_, weight=weight, runtime_gather_output=runtime_gather_output
        )

    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
    )

    state = self._poet_state
    bias = self.bias if not self.skip_bias_add else None

    if (
        self.allreduce_dgrad
        or self.sequence_parallel
        or self.explicit_expert_comm
        or self.disable_grad_reduce
    ):
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    output_parallel = _poetx_core(input_parallel, state, self.weight, bias)

    gather_output = self.gather_output
    if runtime_gather_output is not None:
        gather_output = runtime_gather_output
    if gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def _row_parallel_poetx_forward(self, input_):
    """POET-X fast forward for ``RowParallelLinear``.

    For ``RowParallelLinear`` the math requires applying ``R_out`` to the
    full (TP-reduced) output, not to per-rank partials. ``forward_core``
    applies ``R_out`` right after the linear. At TP=1 that's identical
    (there's no reduction step), and the YAML forces TP=1 for POET-X --
    so we run ``forward_core`` and let the (identity-at-TP=1) reduction
    branch handle the output.

    sequence_parallel=True and explicit_expert_comm=True (the latter is
    standard for MoE routed experts wrapped via --poet-wrap-moe-experts)
    are *not* a problem at TP=1: the corresponding TP comms are no-ops.
    We mirror RowParallelLinear's native output branching so the code is
    semantically aligned for any future TP>1 fix (and so the path matches
    the weight-space ``_row_parallel_poet_forward``).

    Only TP>1 is a real correctness issue (per-rank random permutations
    cannot compose across the row-parallel all-reduce), so that's where
    we hard-fail.
    """
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )

    state = self._poet_state

    # RowParallelLinear doesn't expose ``tp_size``; derive it from the
    # input-dim sharding (matches ``RowParallelLinear.__repr__``).
    tp_size = max(1, self.input_size // max(1, self.input_size_per_partition))
    if tp_size > 1:
        raise NotImplementedError(
            "POET-X fast path on RowParallelLinear currently assumes TP=1. "
            f"Detected TP={tp_size} on {type(self).__name__}. Either set "
            "TP=1 in the YAML or switch to the weight-space "
            "--poet-variant poet path."
        )

    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

    output_parallel = _poetx_core(
        input_parallel,
        state,
        self.weight,
        None if self.skip_bias_add else self.bias,
    )

    # Mirror RowParallelLinear.forward's output branching. At TP=1 all three
    # branches are identity, but keep the dispatch so the code is correct
    # for any future TP>1 lift and matches the weight-space POET variant.
    if self.explicit_expert_comm:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )
    else:
        output_ = reduce_from_tensor_model_parallel_region(
            output_parallel, group=self.tp_group
        )

    if not self.skip_bias_add:
        output = output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


# ---------------------------------------------------------------------------
# Attach / install
# ---------------------------------------------------------------------------


def _name_matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    lower = name.lower()
    return any(pat.lower() in lower for pat in patterns)


def _get_te_linear_classes() -> Tuple[type, ...]:
    """Return the TE parallel-linear classes available in this environment.

    We match on the top two TE base classes that cover all four Megatron-TE
    parallel linears we care about:
      * ``TELinear`` -> base for ``TEColumnParallelLinear`` / ``TERowParallelLinear``
      * ``TELayerNormColumnParallelLinear`` -> inherits directly from
        ``te.pytorch.LayerNormLinear`` and is NOT a subclass of ``TELinear``,
        so it needs its own entry.

    The grouped-GEMM classes (``TEColumnParallelGroupedLinear`` /
    ``TERowParallelGroupedLinear``) are intentionally not listed -- they have
    a stacked-expert weight layout that doesn't match a plain (out, in)
    matrix, and they're already excluded via ``exclude_ancestors`` anyway.

    Returns an empty tuple if TE isn't installed.
    """
    try:
        from megatron.core.extensions.transformer_engine import (
            HAVE_TE,
            TELayerNormColumnParallelLinear,
            TELinear,
        )
    except ImportError:
        return ()
    # The TE class names import even without Transformer Engine (they resolve to
    # ``None`` stubs), so the ImportError guard above never fires under the local
    # impl. Check HAVE_TE so we return an empty tuple instead of ``(None, None)``,
    # which would crash ``isinstance(mod, te_classes)`` in install_poet_in_model.
    if not HAVE_TE:
        return ()
    return (TELinear, TELayerNormColumnParallelLinear)


def _inherit_poet_parallel_attrs(module: nn.Module, state: PoetParallelLinearState) -> None:
    """Make ``oft_R`` follow the wrapped linear's Megatron parallel semantics."""
    weight = getattr(module, "weight", None)
    setattr(state.oft_R, "allreduce", getattr(weight, "allreduce", True))
    setattr(state.oft_R, "sequence_parallel", False)
    setattr(state.oft_R, "tensor_model_parallel", False)


def _poet_uses_expert_data_parallel(module: nn.Module) -> bool:
    """Whether this POET-wrapped module should sync over expert-DP, not dense DP."""
    state = getattr(module, "_poet_state", None)
    if state is not None:
        return not getattr(state.oft_R, "allreduce", True)
    weight = getattr(module, "weight", None)
    return not getattr(weight, "allreduce", True)


def _get_poet_sync_group(module: nn.Module):
    """Return the correct DP process group for this POET-wrapped module."""
    from megatron.core import parallel_state as mpu

    if _poet_uses_expert_data_parallel(module):
        return mpu.get_expert_data_parallel_group()
    return mpu.get_data_parallel_group(with_context_parallel=True)


def _get_poet_sync_group_and_src(module: nn.Module):
    try:
        group = _get_poet_sync_group(module)
        src_rank = torch.distributed.get_global_rank(group, 0)
        return group, src_rank
    except Exception:  # pragma: no cover - tolerate uninitialized mpu in smoke tests
        return None, None


def _remember_group(groups: List[torch.distributed.ProcessGroup], group) -> None:
    if group is not None and all(group is not existing for existing in groups):
        groups.append(group)


def _broadcast_poet_state(module: nn.Module, *, include_weight: bool = False):
    """Broadcast one POET module over its dense-DP or expert-DP replica group."""
    group, src_rank = _get_poet_sync_group_and_src(module)
    if group is None:
        return None

    state = getattr(module, "_poet_state", None)
    if state is None:
        return group

    if include_weight:
        torch.distributed.broadcast(module.weight.data, src=src_rank, group=group)
    torch.distributed.broadcast(state.oft_R.data, src=src_rank, group=group)
    torch.distributed.broadcast(state.perm_in, src=src_rank, group=group)
    torch.distributed.broadcast(state.perm_in_inv, src=src_rank, group=group)
    torch.distributed.broadcast(state.perm_out, src=src_rank, group=group)
    torch.distributed.broadcast(state.perm_out_inv, src=src_rank, group=group)
    return group


def _normalize_weight_(weight: torch.Tensor) -> None:
    """In-place row-normalize ``weight`` so that each row has unit L2 norm.

    This mirrors :func:`poet_torch.utils.replace_linear_with_poet`'s
    ``normalize_weights=True`` behaviour and keeps the singular value spectrum
    of ``W_0`` well-conditioned before POET starts rotating it.
    """
    with torch.no_grad():
        norm = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-8)
        weight.div_(norm)


def _try_attach(
    module: nn.Module,
    module_name: str,
    *,
    kind: str,
    block_size: int,
    normalize_weights: bool,
    exclude_patterns: Sequence[str],
    variant: str = "poet",
    mem_efficient: bool = False,
) -> bool:
    """Attach POET state + patch forward on a single parallel linear.

    ``variant`` is one of:
        "poet"   -- materialize ``W_eff`` per forward (original POET math).
        "poetx"  -- input-centric POET-X_fast path (no W_eff materialization).
    """
    if _name_matches(module_name, exclude_patterns):
        return False
    if getattr(module, "_poet_state", None) is not None:  # already wrapped
        return False
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, nn.Parameter):
        return False

    if kind == "column":
        out_local = getattr(module, "output_size_per_partition", None)
        in_local = getattr(module, "input_size", None)
    elif kind == "row":
        out_local = getattr(module, "output_size", None)
        in_local = getattr(module, "input_size_per_partition", None)
    else:
        return False

    if out_local is None or in_local is None:
        return False
    if out_local % block_size != 0 or in_local % block_size != 0:
        logger.info(
            "POET: skipping %s (kind=%s, out_local=%d, in_local=%d) -- "
            "not divisible by block_size=%d",
            module_name,
            kind,
            out_local,
            in_local,
            block_size,
        )
        return False

    # Freeze the base weight so only oft_R trains.
    weight.requires_grad = False

    # Row-normalize W_0 for a well-conditioned starting spectrum.
    if normalize_weights:
        _normalize_weight_(weight.data)

    state = PoetParallelLinearState(
        in_features_local=in_local,
        out_features_local=out_local,
        block_size=block_size,
        dtype=weight.dtype,
        device=weight.device,
    )
    _inherit_poet_parallel_attrs(module, state)
    # Only the poetx fast path reads this; weight-space "poet" ignores it.
    state.mem_efficient_mode = bool(mem_efficient) and variant == "poetx"
    # Register as a submodule so parameters/buffers show up in named_parameters
    # and are handled correctly by Megatron's checkpointing.
    module.add_module("_poet_state", state)

    # Disable gradient-accumulation fusion for this wrapped linear.
    # With POET, the tensor passed as ``weight`` to the underlying GEMM is
    # ``W_eff`` (a non-leaf Tensor produced from the frozen ``W_0`` and the
    # trainable ``oft_R``). Megatron's ``LinearWithGradAccumulationAndAsync-
    # Communication`` would otherwise:
    #   (1) try to dereference ``W_eff.main_grad`` -- ``AttributeError`` since
    #       main_grad is only attached to real ``nn.Parameter`` leaves by DDP;
    #   (2) even if it didn't crash, it would write the wgrad straight into a
    #       ``main_grad`` buffer and short-circuit autograd, so grads would
    #       never reach ``oft_R``. The real trainable param is ``oft_R``,
    #       whose grad still flows via autograd through ``_compute_effective_weight``.
    if hasattr(module, "gradient_accumulation_fusion"):
        module.gradient_accumulation_fusion = False

    # Monkey-patch forward.
    module._poet_orig_forward = module.forward
    module._poet_variant = variant
    if variant == "poetx":
        fn = _column_parallel_poetx_forward if kind == "column" else _row_parallel_poetx_forward
    elif variant == "poet":
        fn = _column_parallel_poet_forward if kind == "column" else _row_parallel_poet_forward
    else:
        raise ValueError(f"Unknown POET variant: {variant!r} (expected 'poet' or 'poetx')")
    module.forward = types.MethodType(fn, module)

    logger.info(
        "POET[%s]: wrapped %s [%s] (out_local=%d, in_local=%d, block_size=%d, "
        "n_blocks=%d, oft_R params=%d)",
        variant,
        module_name,
        kind,
        out_local,
        in_local,
        block_size,
        state.num_out_blocks + state.num_in_blocks,
        state.oft_R.numel(),
    )
    return True


def _try_attach_te(
    module: nn.Module,
    module_name: str,
    *,
    block_size: int,
    normalize_weights: bool,
    exclude_patterns: Sequence[str],
) -> bool:
    """Attach POET state + weight-swap forward on a TE parallel linear.

    TE linears store ``self.weight`` with shape ``(out_local, in_local)`` on
    this TP rank regardless of whether they are column / row parallel or
    LN-fused, so we use ``weight.shape`` as the source of truth for local
    dims and don't need a separate ``kind`` dispatch.
    """
    if _name_matches(module_name, exclude_patterns):
        return False
    if getattr(module, "_poet_state", None) is not None:
        return False
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        return False

    out_local, in_local = int(weight.shape[0]), int(weight.shape[1])
    if out_local % block_size != 0 or in_local % block_size != 0:
        logger.info(
            "POET[TE]: skipping %s (%s, out_local=%d, in_local=%d) -- "
            "not divisible by block_size=%d",
            module_name,
            type(module).__name__,
            out_local,
            in_local,
            block_size,
        )
        return False

    # FP8 + POET weight-swap is unsafe: TE caches an FP8-quantized copy of the
    # weight and would keep reusing it across forwards even though W_eff
    # changes every step. Refuse to wrap in that case.
    fp8 = getattr(getattr(module, "config", None), "fp8", None)
    if fp8 not in (None, "", False):
        raise RuntimeError(
            f"POET[TE]: cannot wrap {module_name} -- config.fp8={fp8!r}. "
            "TE caches FP8-quantized weights, which conflicts with POET's "
            "per-forward W_eff. Disable FP8 or exclude this layer."
        )

    weight.requires_grad = False
    if normalize_weights:
        _normalize_weight_(weight.data)

    state = PoetParallelLinearState(
        in_features_local=in_local,
        out_features_local=out_local,
        block_size=block_size,
        dtype=weight.dtype,
        device=weight.device,
    )
    _inherit_poet_parallel_attrs(module, state)
    module.add_module("_poet_state", state)

    # Force TE to rebuild any weight-derived state every forward. With
    # ``disable_parameter_transpose_cache=False``, TE stashes a column-major
    # copy of ``weight`` after the first microbatch and reuses it -- which
    # would be silently stale because we rebind ``weight`` to ``W_eff`` on
    # every forward. Same argument for FP8 (already refused above).
    if hasattr(module, "disable_parameter_transpose_cache"):
        module.disable_parameter_transpose_cache = True
    if hasattr(module, "is_first_microbatch"):
        module.is_first_microbatch = True

    # Disable TE's fused wgrad accumulation on this linear. TE would otherwise
    # (a) capture ``W_eff`` in a closure and try ``W_eff.main_grad`` in backward
    # (``AttributeError`` -- main_grad lives on nn.Parameter leaves only), and
    # (b) write wgrad directly into that main_grad buffer, bypassing autograd,
    # so grads would never flow back through ``_compute_effective_weight`` to
    # the real trainable parameter ``oft_R``. See TE ``module/linear.py``
    # (``ctx.main_grad_func = lambda: weight.main_grad`` at L435 and
    # ``out=main_grad if fuse_wgrad_accumulation else None`` at L831).
    if hasattr(module, "fuse_wgrad_accumulation"):
        module.fuse_wgrad_accumulation = False

    module._poet_orig_forward = module.forward
    module._poet_variant = "poet"  # weight-space; poetx not applicable for TE LN-fused
    module.forward = types.MethodType(_te_poet_weight_swap_forward, module)

    logger.info(
        "POET[TE/poet]: wrapped %s [%s] (out_local=%d, in_local=%d, block_size=%d, "
        "n_blocks=%d, oft_R params=%d)",
        module_name,
        type(module).__name__,
        out_local,
        in_local,
        block_size,
        state.num_out_blocks + state.num_in_blocks,
        state.oft_R.numel(),
    )
    return True


def _is_inside_excluded_ancestor(
    model: nn.Module, module_name: str, ancestor_patterns: Sequence[str]
) -> bool:
    if not ancestor_patterns:
        return False
    tokens = module_name.split(".")
    for i in range(1, len(tokens) + 1):
        sub = ".".join(tokens[:i])
        if _name_matches(sub, ancestor_patterns):
            return True
    return False


def install_poet_in_model(
    model: nn.Module,
    *,
    block_size: int = 256,
    exclude_modules: Optional[Sequence[str]] = None,
    exclude_ancestors: Optional[Sequence[str]] = None,
    normalize_weights: bool = True,
    variant: str = "poet",
    mem_efficient: bool = False,
) -> int:
    """Install POET reparameterization on eligible parallel linears.

    Args:
        model: the Megatron model (anywhere in the nn.Module tree).
        block_size: POET block size.
        exclude_modules: substrings to exclude at the leaf-module level.
            Defaults to ``("lm_head", "output_layer", "embedding",
            "word_embeddings", "router", "gate", "mtp")`` -- i.e. small/output
            layers that don't benefit from POET.
        exclude_ancestors: substrings that, if present anywhere in a module's
            qualified name, cause the module and its descendants to be
            skipped. Defaults to ``("experts", "local_experts",
            "grouped_mlp", "te_grouped_mlp")`` so MoE grouped-GEMM experts are
            left alone (their weight layout doesn't match plain
            Column/Row-parallel linears).
        normalize_weights: if True, row-normalize W_0 before starting POET.
        variant: "poet" (materialize W_eff) or "poetx" (input-centric,
            activation-space; recommended).
        mem_efficient: if True and variant="poetx", route the forward through
            the ``chain_layer_x_checkpoint_mem_o2`` Triton kernel so that the
            permute + R_in + linear + R_out + permute chain recomputes its
            internal activations in backward (trades extra compute for roughly
            half the activation memory of the chain). No effect for "poet".

    Returns:
        Number of layers wrapped.
    """
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    if exclude_modules is None:
        exclude_modules = (
            "lm_head",
            "output_layer",
            "embedding",
            "word_embeddings",
            "router",
            "gate",
            "mtp",
        )
    if exclude_ancestors is None:
        # Note: we deliberately do *not* exclude "experts" wholesale, because
        # shared experts (``mlp.shared_experts.*``) are regular
        # Column/Row-parallel linears and benefit from POET. We only skip the
        # grouped-GEMM / per-expert MoE weights whose layouts differ.
        exclude_ancestors = (
            "local_experts",
            "grouped_mlp",
            "te_grouped_mlp",
            ".experts.",  # matches ``mlp.experts.`` but not ``shared_experts``.
        )

    if variant not in ("poet", "poetx"):
        raise ValueError(f"Unknown POET variant: {variant!r} (expected 'poet' or 'poetx')")

    te_classes = _get_te_linear_classes()

    n_wrapped = 0
    for name, mod in model.named_modules():
        if _is_inside_excluded_ancestor(model, name, exclude_ancestors):
            continue
        if isinstance(mod, ColumnParallelLinear):
            if _try_attach(
                mod,
                name,
                kind="column",
                block_size=block_size,
                normalize_weights=normalize_weights,
                exclude_patterns=exclude_modules,
                variant=variant,
                mem_efficient=mem_efficient,
            ):
                n_wrapped += 1
        elif isinstance(mod, RowParallelLinear):
            if _try_attach(
                mod,
                name,
                kind="row",
                block_size=block_size,
                normalize_weights=normalize_weights,
                exclude_patterns=exclude_modules,
                variant=variant,
                mem_efficient=mem_efficient,
            ):
                n_wrapped += 1
        elif te_classes and isinstance(mod, te_classes):
            # TE path always uses the weight-space variant (activation-space
            # poetx can't be applied cleanly through TE's fused LN+Linear).
            if variant != "poet":
                logger.warning(
                    "POET[TE]: forcing variant='poet' for %s (%s); "
                    "activation-space poetx is not supported on TE LN-fused linears.",
                    name,
                    type(mod).__name__,
                )
            if _try_attach_te(
                mod,
                name,
                block_size=block_size,
                normalize_weights=normalize_weights,
                exclude_patterns=exclude_modules,
            ):
                n_wrapped += 1

    if n_wrapped == 0:
        seen = {}
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if "Linear" in cls or "linear" in name.split(".")[-1]:
                seen[cls] = seen.get(cls, 0) + 1
        inv = ", ".join(f"{k}:{v}" for k, v in sorted(seen.items())[:12])
        raise RuntimeError(
            f"POET[{variant}]: no parallel linear layers were wrapped. "
            f"Eligible classes are Megatron's native ColumnParallelLinear / "
            f"RowParallelLinear, and (if TE is installed) TELinear / "
            f"TEColumnParallelLinear / TERowParallelLinear / "
            f"TELayerNormColumnParallelLinear. Linear-like modules present "
            f"in the model: {inv}. Verify block_size={block_size} divides "
            f"every wrapped linear's local in/out dims."
        )

    # ----------------------------------------------------------------------
    # Broadcast perm_* buffers from DP rank 0 to all other DP ranks.
    #
    # Why this is needed:
    #   * ``perm_in`` / ``perm_out`` are registered as nn.Module **buffers**,
    #     not parameters. Megatron's DDP only synchronizes parameters
    #     (``broadcast_params``, and even that only fires when
    #     ``--data-parallel-random-init`` is set). Buffers are NOT
    #     auto-synced.
    #   * The values are produced by ``torch.randperm`` against the default
    #     cuda RNG state. Megatron's ``model_parallel_cuda_manual_seed``
    #     does start every DP rank with the same default state, but staying
    #     in lock-step until ``install_poet_in_model`` runs depends on every
    #     model-init code path consuming the default RNG the same number of
    #     times on every DP rank. That's an implicit contract, not a
    #     guarantee -- e.g. any future MoE / router init that touches the
    #     default state on a per-rank-conditional path would silently
    #     desynchronize the perms.
    #   * If perms diverge across DP ranks, every rank computes a *different*
    #     ``W_eff = R_out @ W_0 @ R_in`` (different permutation overlay),
    #     so DDP grad averaging averages gradients of different functions
    #     -- the optimizer no longer minimizes a single coherent loss.
    #
    # The merge code already broadcasts perms within the DP group at every
    # ``--poet-merge-interval`` (see ``merge_all_poet_layers``), but the
    # first ``merge_interval`` steps run *before* the first merge and thus
    # before the first broadcast. We close that window here.
    #
    # No-op when distributed isn't initialized (CPU smoke tests).
    # ----------------------------------------------------------------------
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_groups = []
        for _, mod in model.named_modules():
            state = getattr(mod, "_poet_state", None)
            if state is None:
                continue
            group = _broadcast_poet_state(mod, include_weight=False)
            _remember_group(sync_groups, group)
        for group in sync_groups:
            torch.distributed.barrier(group=group)

    logger.info("POET[%s]: total wrapped layers = %d", variant, n_wrapped)

    return n_wrapped


# ---------------------------------------------------------------------------
# Merge-then-reinitialize
# ---------------------------------------------------------------------------


@torch.no_grad()
def _merge_one_layer(module: nn.Module) -> None:
    """Absorb R_out / R_in into the local base weight and reset oft_R.

    Each TP rank merges its own shard independently (permutations are rank-local).
    """
    from poet_torch.core.ops import block_diag_lr_matmul

    state: PoetParallelLinearState = module._poet_state
    weight = module.weight  # (out_local, in_local)

    R_out, R_in = _compute_R_out_R_in(state, weight.dtype)

    tmp = weight.data.t()
    tmp = block_diag_lr_matmul(R_in, tmp, R_out)
    tmp = tmp.index_select(0, state.perm_in.long())
    tmp = tmp.index_select(1, state.perm_out.long())
    W_new = tmp.t()

    # Fresh permutations for the next merge cycle.
    state._reinit_permutations()

    # Apply inverse of the *new* permutations to the merged weight so that
    # the next forward reproduces the same effective weight.
    W_new = W_new.index_select(0, state.perm_out_inv.long()).index_select(
        1, state.perm_in_inv.long()
    )

    weight.data.copy_(W_new)
    state.oft_R.data.zero_()


@torch.no_grad()
def merge_all_poet_layers(
    model: nn.Module,
    *,
    step: int,
    merge_interval: int,
    reset_optimizer_state_for: Optional[List[torch.nn.Parameter]] = None,
    optimizer=None,
) -> bool:
    """Run merge-then-reinitialize on every POET-wrapped layer if it's due.

    Args:
        model: (possibly DDP-wrapped) model.
        step: current global step.
        merge_interval: merge every ``merge_interval`` steps.
        reset_optimizer_state_for: optional explicit list of POET params whose
            optimizer state should be reset after the merge. Defaults to all
            merged ``oft_R`` params.
        optimizer: Megatron optimizer wrapper for the current training step.
            When provided, we refresh optimizer-side fp32 main params from the
            merged model params and reset Adam moments for merged ``oft_R``.

    Returns:
        True if a merge was performed this call.
    """
    if merge_interval <= 0 or step <= 0 or (step % merge_interval != 0):
        return False

    # ------------------------------------------------------------------
    # Force-sync model params before doing anything else.
    #
    # Why: with --use-distributed-optimizer + --overlap-param-gather, the
    # post-step all-gather (which writes the just-updated fp32 main_param
    # back into the bf16 ``param.data`` view of the param buffer) is *not*
    # dispatched at the end of optimizer.step(); it is deferred to the
    # next iter's optimizer.zero_grad() / forward pre-hook
    # (see DistributedOptimizer.step_with_ready_grads -- the
    # ``not overlap_param_gather`` branch is the one that calls
    # ``start_param_sync`` synchronously).
    #
    # Consequence at the moment ``merge_all_poet_layers`` runs:
    #   1) ``state.oft_R.data`` is the value from the *previous* iter's
    #      gather, i.e. it does NOT include the just-completed step's
    #      update. Computing R_in / R_out from that value and merging it
    #      into W effectively throws the latest oft_R update away. (#3)
    #   2) ``optimizer.reload_model_params()`` (called below to refresh
    #      main_param after we zero oft_R.data) iterates over *all*
    #      param groups and copies model_param.data -> main_param. For
    #      every non-oft_R trainable param (embeddings, MoE experts,
    #      router, biases, norms) model_param.data is still the pre-step
    #      value, so reload silently REVERTS the optimizer step on those
    #      params. (#2)
    #
    # Calling start_param_sync(force_sync=True) here issues a synchronous
    # all-gather, so model_param.data == post-step main_param for every
    # trainable param when we proceed. After that:
    #   - Reading oft_R.data sees the latest value (fixes #3).
    #   - reload_model_params copies the current (post-step) model_param
    #     into main_param, which is a no-op for non-oft_R params and the
    #     desired zero for oft_R after our merge clears its .data
    #     (fixes #2).
    #
    # We tolerate models that are not DDP-wrapped (CPU smoke tests, etc.)
    # by falling back silently if the chunk has no ``start_param_sync``.
    # ------------------------------------------------------------------
    _force_post_step_param_sync(model)

    # Only rank 0 in each replica group does the merge to avoid permutation
    # divergence across replicas; TP ranks each do their own shard.  Dense
    # layers use the ordinary DP+CP group, while routed MoE experts must use
    # the expert-DP group because different EP ranks own different experts.
    #
    # In practice for Megatron with `--use-distributed-optimizer` and DDP the
    # frozen W_0 is identical across that parameter's replica group, and
    # oft_R is reduced over the same group. The only non-deterministic piece
    # is the new randperm; we therefore broadcast the post-merge state from
    # rank 0 within each module's dense-DP or expert-DP group.
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    # Why the set_stance + _dynamo.reset below:
    #
    # ``forward_core`` (poet_torch) is ``@torch.compile(fullgraph=True)``.
    # Dynamo compiles one entry per unique input-tensor-profile and installs
    # guards that include the tensor ``_version`` counter. ``_merge_one_layer``
    # uses ``.copy_()`` to update ``weight.data`` and the ``perm_*`` buffers
    # in-place, which bumps ``_version`` on every single wrapped linear.
    # On the next forward every guard fails -> Dynamo retraces. With N
    # wrapped linears x (merge_events + 1) distinct versions, the default
    # ``cache_size_limit=8`` overflows almost immediately on multi-layer
    # MoE models and, because fullgraph=True sets ``one_graph=True``, we
    # get ``FailOnRecompileLimitHit`` instead of a graceful fallback.
    #
    # The merge invalidates every old compiled artifact anyway (guards would
    # all fail), so the right thing is to drop them and re-trace cleanly on
    # the next forward. ``set_stance("eager_then_compile")`` additionally
    # prevents Dynamo from trying to trace the merge itself (matches the
    # upstream ``poet`` ``merge_and_reinitialize`` wrapping).
    try:
        _stance_ctx = torch.compiler.set_stance("eager_then_compile")
    except Exception:  # pragma: no cover - older torch without set_stance
        import contextlib
        _stance_ctx = contextlib.nullcontext()

    with _stance_ctx:
        model_chunks = _unwrap_model_chunks(model)
        merged_poet_params = []
        modules_to_broadcast = []
        for chunk in model_chunks:
            for _, mod in chunk.named_modules():
                state = getattr(mod, "_poet_state", None)
                if state is None:
                    continue
                merged_poet_params.append(state.oft_R)
                modules_to_broadcast.append(mod)

                sync_group = None
                sync_rank = 0
                if is_dist:
                    sync_group, _ = _get_poet_sync_group_and_src(mod)
                    if sync_group is not None:
                        sync_rank = torch.distributed.get_rank(sync_group)

                if sync_rank == 0:
                    _merge_one_layer(mod)

        if is_dist:
            sync_groups = []
            for mod in modules_to_broadcast:
                group = _broadcast_poet_state(mod, include_weight=True)
                _remember_group(sync_groups, group)
            for group in sync_groups:
                torch.distributed.barrier(group=group)

    # Drop the now-stale Dynamo cache so the next forward recompiles from
    # scratch instead of failing guard checks N times in a row.
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
    except Exception:  # pragma: no cover
        pass

    params_to_reset = reset_optimizer_state_for
    if params_to_reset is None:
        params_to_reset = merged_poet_params
    if optimizer is not None and params_to_reset:
        _sync_and_reset_poet_optimizer_state(optimizer, params_to_reset)

    return True


def _zero_optimizer_state_value(state, key):
    value = state[key]
    if isinstance(value, torch.Tensor):
        value.zero_()
    elif isinstance(value, (int, float)):
        state[key] = type(value)(0)


def _reset_optimizer_state_for_model_params(optimizer, model_params: Sequence[torch.nn.Parameter]) -> None:
    inner_optimizer = getattr(optimizer, "optimizer", None)
    if inner_optimizer is None:
        return
    state = getattr(inner_optimizer, "state", None)
    if state is None:
        return

    for model_param in model_params:
        main_param = getattr(model_param, "main_param", model_param)
        if main_param not in state:
            continue
        param_state = state[main_param]
        for key in list(param_state.keys()):
            _zero_optimizer_state_value(param_state, key)


def _sync_and_reset_poet_optimizer_state(
    optimizer, model_params: Sequence[torch.nn.Parameter]
) -> None:
    if optimizer is None or getattr(optimizer, "is_stub_optimizer", False):
        return

    if hasattr(optimizer, "chained_optimizers"):
        for sub_optimizer in optimizer.chained_optimizers:
            _sync_and_reset_poet_optimizer_state(sub_optimizer, model_params)
        return

    optimizer.reload_model_params()
    _reset_optimizer_state_for_model_params(optimizer, model_params)


def _unwrap_model_chunks(model) -> List[nn.Module]:
    """Peel off wrappers and always return a flat list of model chunks."""
    if isinstance(model, (list, tuple)):
        chunks: List[nn.Module] = []
        for item in model:
            chunks.extend(_unwrap_model_chunks(item))
        return chunks
    return [_unwrap_single(model)]


def _unwrap_single(m):
    for _ in range(6):
        if hasattr(m, "module"):
            m = m.module
        else:
            break
    return m


def _iter_outer_model_chunks(model):
    """Yield the outer (DDP-wrapped) chunks from whatever ``train_step`` passes
    in -- which can be either a single chunk or a (possibly nested) list."""
    if isinstance(model, (list, tuple)):
        for item in model:
            yield from _iter_outer_model_chunks(item)
        return
    yield model


def _force_post_step_param_sync(model) -> None:
    """Synchronously refresh ``param.data`` for every trainable param.

    With ``--use-distributed-optimizer + --overlap-param-gather`` Megatron
    defers the post-step all-gather to the next iter's
    ``optimizer.zero_grad()``. We need the latest values *now* so the merge
    sees the just-stepped oft_R and so ``reload_model_params`` doesn't
    revert other params. ``start_param_sync(force_sync=True)`` issues a
    synchronous all-gather (see ``DistributedDataParallel.start_param_sync``
    and ``_ParamAndGradBucketGroup.start_param_sync``). It also waits on a
    pending async handle if one happens to be in flight.

    No-op when the chunk isn't DDP-wrapped (e.g. CPU smoke tests) or when
    distributed-optimizer mode is off.
    """
    for chunk in _iter_outer_model_chunks(model):
        sync_fn = getattr(chunk, "start_param_sync", None)
        if sync_fn is None:
            continue
        try:
            sync_fn(force_sync=True)
        except AssertionError:
            # ``_ParamAndGradBucketGroup.start_param_sync`` asserts
            # ``use_distributed_optimizer``. If the user runs without it the
            # post-step ``param.data`` is already up-to-date and there is
            # nothing for us to do.
            continue


# ---------------------------------------------------------------------------
# CLI args + config assembly
# ---------------------------------------------------------------------------


def add_poet_args(parser):
    """Register POET CLI arguments on a Megatron argparser."""
    group = parser.add_argument_group(title="POET (Orthogonal Equivalence Transformation)")
    group.add_argument(
        "--use-poet",
        action="store_true",
        help="Enable POET / POET-X reparameterized training on parallel linears.",
    )
    group.add_argument(
        "--poet-variant",
        type=str,
        choices=["poet", "poetx"],
        default="poet",
        help=(
            "Which forward implementation to use: 'poet' materializes W_eff "
            "per forward (original POET), 'poetx' uses the input-centric "
            "POET-X_fast path which avoids W_eff materialization and therefore "
            "saves activation/parameter memory during training."
        ),
    )
    group.add_argument(
        "--poet-block-size",
        type=int,
        default=256,
        help="POET block size. Local in/out dims of every wrapped linear must be divisible by this.",
    )
    group.add_argument(
        "--poet-merge-interval",
        type=int,
        default=200,
        help="Merge-then-reinitialize every N optimizer steps.",
    )
    group.add_argument(
        "--poet-mem-efficient",
        action="store_true",
        help="Use the memory-efficient POET-X path (requires Triton).",
    )
    group.add_argument(
        "--poet-quantize",
        action="store_true",
        help="Use POET-XQ (INT8 quantized base weights, requires Triton).",
    )
    group.add_argument(
        "--poet-no-normalize-weights",
        dest="poet_normalize_weights",
        action="store_false",
        help="Disable row-normalization of W_0 at install time.",
    )
    parser.set_defaults(poet_normalize_weights=True)
    # ------------------------------------------------------------------
    # Optimizer-side POET hygiene. These match the reference POETAdamW
    # defaults (poet_torch/config.py + poet_torch/optim/adamw.py):
    #   * oft_R gets weight_decay=0 (wd pulls oft_R back toward zero, i.e.
    #     R back toward identity, which fights POET throughout training).
    #   * oft_R uses a scaled-down LR; reference is
    #     ``poet_lr * poet_scale / base_lr = 5e-4 * 0.5 / 1e-3 = 0.25``.
    # We expose both as CLI so you can still sweep them without editing
    # code, but the defaults reproduce the paper setup.
    # ------------------------------------------------------------------
    group.add_argument(
        "--poet-oft-lr-scale",
        type=float,
        default=0.25,
        help=(
            "LR multiplier applied to oft_R parameters relative to the "
            "global --lr. Defaults to 0.25 to reproduce the reference "
            "POETAdamW ratio (poet_lr * poet_scale / base_lr). Set to 1.0 "
            "to disable scaling."
        ),
    )
    group.add_argument(
        "--poet-oft-weight-decay",
        dest="poet_oft_weight_decay",
        action="store_true",
        help=(
            "Apply the global --weight-decay to oft_R. Off by default; "
            "reference training uses weight_decay=0 for POET params."
        ),
    )
    parser.set_defaults(poet_oft_weight_decay=False)
    group.add_argument(
        "--poet-exclude-modules",
        type=str,
        nargs="*",
        default=None,
        help="Extra leaf-module name substrings to exclude from POET wrapping.",
    )
    group.add_argument(
        "--poet-exclude-ancestors",
        type=str,
        nargs="*",
        default=None,
        help="Extra ancestor name substrings to exclude (skips all descendants).",
    )
    group.add_argument(
        "--poet-wrap-moe-experts",
        action="store_true",
        help=(
            "Wrap routed MoE expert MLP parallel linears (SequentialMLP) with POET. "
            "Incompatible with --moe-grouped-gemm / GroupedMLP fused expert weights "
            "(disable grouped GEMM so experts use Megatron TP linears per expert). "
            "Also verify --poet-block-size divides moe_ffn_hidden_size "
            "(e.g. moe_ffn_hidden_size=896 needs block_size <= 128, not 256)."
        ),
    )
    return parser


def get_poet_config_from_args(args):
    """Build a ``POETConfig`` (or ``QPOETConfig``) from Megatron args."""
    from poet_torch import POETConfig, QPOETConfig

    cls = QPOETConfig if getattr(args, "poet_quantize", False) else POETConfig
    return cls(
        block_size=args.poet_block_size,
        mem_efficient_mode=args.poet_mem_efficient,
        merge_interval=args.poet_merge_interval,
        # Megatron drives the LR schedule; we don't use POET's own lrs.
        poet_lr=args.lr if hasattr(args, "lr") else 1e-3,
        base_lr=args.lr if hasattr(args, "lr") else 1e-3,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )


def _is_oft_r(name: str) -> bool:
    """Name predicate for POET's trainable Cayley vectors."""
    return name.endswith("oft_R") or name.endswith("._poet_state.oft_R")


def install_poet_optimizer_hook() -> None:
    """Patch ``megatron.training.training.setup_model_and_optimizer`` so that
    POET's trainable ``oft_R`` params are placed in a separate param group
    with ``weight_decay=0`` and (optionally) a scaled-down LR.

    Why this is needed: by default Megatron's ``_get_param_groups`` treats
    every 2D parameter as a regular weight -> ``wd_mult=1, lr_mult=1``. The
    reference POET optimizer (see ``poet_torch/optim/adamw.py``) instead
    sets ``weight_decay=0`` for POET params and applies
    ``lr_eff = poet_lr * poet_scale`` (≈ 0.25x the base LR in the upstream
    defaults). Leaving ``oft_R`` in the default group pulls it toward zero
    every step, which directly fights POET's ability to rotate weights
    away from identity. This hook restores the paper's behavior.

    The hook is opt-in: it only rewrites the conds when ``--use-poet`` is
    on, so non-POET runs are completely unaffected. It also preserves
    Megatron's default bias/norm/embedding weight-decay handling -- we
    only *add* the oft_R exclusion on top.

    Safe to call multiple times; guarded by an attribute on the target
    module so repeated calls are no-ops.
    """
    from megatron.training import training as _training_module

    if getattr(_training_module, "_poet_optimizer_hook_installed", False):
        return

    original_setup_model_and_optimizer = _training_module.setup_model_and_optimizer

    def _patched_setup_model_and_optimizer(
        model_provider_func,
        model_type,
        no_wd_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        checkpointing_context=None,
    ):
        from megatron.training import get_args

        args = get_args()
        if not getattr(args, "use_poet", False):
            return original_setup_model_and_optimizer(
                model_provider_func,
                model_type,
                no_wd_decay_cond=no_wd_decay_cond,
                scale_lr_cond=scale_lr_cond,
                lr_mult=lr_mult,
                checkpointing_context=checkpointing_context,
            )

        apply_wd_to_oft_r = bool(getattr(args, "poet_oft_weight_decay", False))
        oft_lr_scale = float(getattr(args, "poet_oft_lr_scale", 1.0))

        # Replicate Megatron's default no-wd rule so we can compose the
        # oft_R exclusion on top without losing bias/norm/embedding behavior.
        # This mirrors megatron/core/optimizer/__init__.py, L106-114.
        default_skip_embedding_weight_decay = (
            getattr(args, "embedding_init_method_std", None) is not None
        )

        user_no_wd = no_wd_decay_cond
        user_scale_lr = scale_lr_cond
        user_lr_mult = lr_mult

        def _poet_no_wd_decay_cond(name: str, param) -> bool:
            if not apply_wd_to_oft_r and _is_oft_r(name):
                return True
            if user_no_wd is not None:
                return user_no_wd(name, param)
            return (
                name.endswith(".bias")
                or len(param.shape) == 1
                or (default_skip_embedding_weight_decay and "embedding" in name)
            )

        def _poet_scale_lr_cond(name: str, param) -> bool:
            if oft_lr_scale != 1.0 and _is_oft_r(name):
                return True
            if user_scale_lr is not None:
                return user_scale_lr(name, param)
            return False

        # If the upstream caller also requested an lr_mult != 1.0 we cannot
        # honor both with a single ``lr_mult`` value (Megatron only supports
        # one multiplier per ``_get_param_groups`` call). Prefer POET's
        # scale since that's the whole point of this hook, but make the
        # conflict visible.
        effective_lr_mult = oft_lr_scale
        if user_lr_mult != 1.0 and user_lr_mult != oft_lr_scale:
            logger.warning(
                "POET: overriding caller lr_mult=%.4f with --poet-oft-lr-scale=%.4f. "
                "Megatron only supports one global lr_mult; if you need "
                "distinct scalings, extend _get_param_groups.",
                user_lr_mult,
                oft_lr_scale,
            )

        return original_setup_model_and_optimizer(
            model_provider_func,
            model_type,
            no_wd_decay_cond=_poet_no_wd_decay_cond,
            scale_lr_cond=_poet_scale_lr_cond,
            lr_mult=effective_lr_mult,
            checkpointing_context=checkpointing_context,
        )

    _training_module.setup_model_and_optimizer = _patched_setup_model_and_optimizer
    _training_module._poet_optimizer_hook_installed = True


def get_poet_param_names(model: nn.Module) -> List[str]:
    """Return the qualified names of all trainable POET params (``oft_R``)."""
    names: List[str] = []
    for name, p in model.named_parameters():
        if name.endswith("oft_R") and p.requires_grad:
            names.append(name)
    return names
