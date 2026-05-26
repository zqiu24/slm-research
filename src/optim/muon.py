"""Thin adapter around Megatron-Core's pinned Muon optimizer.

Muon is already present in `third_party/Megatron-LM` at the slm-research pin.
This module gives slm-research a stable import surface without copying Muon
implementation code into `src/`.
"""

from __future__ import annotations

from typing import Any


def get_megatron_muon_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Any = None,
) -> Any:
    from megatron.core.optimizer.muon import (
        get_megatron_muon_optimizer as _get_megatron_muon_optimizer,
    )

    return _get_megatron_muon_optimizer(
        config=config,
        model_chunks=model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        layer_wise_distributed_optimizer=layer_wise_distributed_optimizer,
        pg_collection=pg_collection,
    )
