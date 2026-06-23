"""Optimizer dispatch layer.

Each optimizer family lives in its own module (adam: torch.optim,
muon: muon.py, poet: poet.py). ``get_optimizer(cfg, params, mcore_cfg)``
selects one by ``cfg.kind``; ``mcore_cfg`` is forwarded for builders that
need the Megatron OptimizerConfig (mixed-precision wrapper construction).

POET is special: it needs the assembled model chunks (not bare params)
because it has to walk the model for the linear/non-linear param split.
Callers must use ``src.optim.poet.get_megatron_poet_optimizer`` directly
when ``cfg.kind == "poet"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

_VALID_KINDS = frozenset({"adam", "adamw", "sgd", "muon", "poet"})


@dataclass
class OptimizerCfg:
    """Configuration for the optimizer family.

    Loaded from experiment YAML; passed to ``get_optimizer``.
    """

    kind: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # POET-specific (zero-valued for non-POET runs; harmless to carry).
    poet_merge_period: int = 0
    poet_scale: float = 1.0
    poet_block_size: int = 256
    poet_init_type: str = "normalized"
    poet_mup_alpha: float = 1.0
    poet_init_scale: float = 1.0

    # Muon-specific (carried for slm-research API symmetry; consumed by
    # src.optim.muon which delegates to Megatron-Core's pinned Muon builder).
    muon_momentum: float = 0.95
    muon_num_ns_steps: int = 5
    muon_scale_mode: str = "spectral"
    muon_tp_mode: str = "blockwise"
    muon_extra_scale_factor: float = 1.0
    muon_coefficient_type: str = "quintic"
    muon_scalar_optimizer: str = "adam"

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown optimizer kind {self.kind!r}; " f"valid: {sorted(_VALID_KINDS)}"
            )


def get_optimizer(
    cfg: OptimizerCfg,
    params: Iterable[torch.nn.Parameter],
    mcore_cfg: Any | None = None,
) -> torch.optim.Optimizer:
    """Dispatch on ``cfg.kind`` to the per-family builder.

    For ``poet``, this raises — POET requires model chunks (not bare params)
    because it has to walk the model for the linear/non-linear split.
    Callers must use ``src.optim.poet.get_megatron_poet_optimizer`` directly.
    """
    if cfg.kind == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if cfg.kind == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if cfg.kind == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.kind == "muon":
        # muon.py not yet implemented; placeholder for the dispatcher contract.
        from src.optim.muon import get_muon_optimizer

        return get_muon_optimizer(cfg, params, mcore_cfg)
    if cfg.kind == "poet":
        raise ValueError(
            "POET optimizer needs the assembled model chunks, not bare "
            "params. Use src.optim.poet.get_megatron_poet_optimizer."
        )
    raise ValueError(f"unknown optimizer kind {cfg.kind!r}")


__all__ = ["OptimizerCfg", "get_optimizer"]
