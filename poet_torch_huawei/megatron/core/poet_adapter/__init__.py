"""Megatron integration for POET / POET-X.

This package adapts the POET (and POET-X) reparameterization from
https://github.com/Sphere-AI-Lab/poet to both Megatron's native parallel
linears (``ColumnParallelLinear`` / ``RowParallelLinear``) and
Transformer-Engine's parallel linears (``TELinear`` / ``TEColumnParallelLinear``
/ ``TERowParallelLinear`` / ``TELayerNormColumnParallelLinear``). TE linears
go through a weight-space forward that rebinds ``self.weight`` to ``W_eff``
so TE's fused LN + GEMM + TP-comm path is preserved.

Key entry points:
    * :func:`install_poet_in_model` -- traverse a Megatron model and replace
      the ``forward`` pass of eligible parallel linears with a POET-parameterized
      forward (``W_eff = R_out @ W_0 @ R_in`` with permutations). Base weights
      are frozen; only the Cayley-parameterized ``oft_R`` parameters train.
    * :func:`merge_all_poet_layers` -- run the periodic
      ``merge-then-reinitialize`` step (absorb R into W and draw fresh
      permutations / reset oft_R).
    * :func:`add_poet_args` -- register CLI arguments with Megatron's argparser.
    * :func:`get_poet_config_from_args` -- build a :class:`POETConfig` /
      :class:`QPOETConfig` from parsed Megatron args.

The underlying POET math (Cayley-Neumann transform, block-diagonal rotation,
permutations, optional Triton kernels) lives in the vendored ``poet_torch``
package at the Megatron-LM repo root. This adapter only adds the Megatron
plumbing around it.
"""

from .adapter import (
    PoetParallelLinearState,
    add_poet_args,
    get_poet_config_from_args,
    get_poet_param_names,
    install_poet_in_model,
    install_poet_optimizer_hook,
    merge_all_poet_layers,
)

__all__ = [
    "PoetParallelLinearState",
    "add_poet_args",
    "get_poet_config_from_args",
    "get_poet_param_names",
    "install_poet_in_model",
    "install_poet_optimizer_hook",
    "merge_all_poet_layers",
]
