"""Builder for the ``muon_kimi`` optimizer.

Wraps the vendored single-process Kimi Muon (``src/optim/_kimi_muon.py``) in
Megatron's optimizer machinery. Single-GPU dev scope only: raises on
tensor-parallel / distributed-optimizer / fp16, none of which the vendored
optimizer supports. Reached via ``src/patches/muon_kimi_optimizer_setup.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class _StripUseMuonShardingMixin:
    """Make the Kimi Muon optimizer safe to dist-checkpoint with Megatron.

    The vendored ``Muon`` stores a per-param boolean ``use_muon`` flag in
    ``optimizer.state`` (set in ``Muon.__init__`` to route each param to the Muon
    or internal-AdamW update). Megatron's ``sharded_state_dict`` runs *every*
    per-param optimizer-state value through ``make_sharded_optimizer_tensor``
    (which assumes tensors) and only excludes ``step`` — so the ``use_muon`` bool
    raises ``AttributeError: 'bool' object has no attribute 'shape'`` during the
    end-of-training ``save_checkpoint`` (fired at the last step because
    ``save_interval`` is large but ``iteration % save_interval != 0``). That crash
    also aborts the post-training validation, so the final-step eval is never
    logged — the muon_kimi val curve stops one eval-interval short of the last
    step while adam/POET reach it.

    Fix: drop ``use_muon`` from the optimizer state while the sharded state dict is
    built, then restore it. ``use_muon`` is re-derived from param routing in
    ``Muon.__init__`` on load (see ``init_state_fn`` below), so omitting it from
    the checkpoint is lossless.
    """

    def sharded_state_dict(self, *args, **kwargs):
        stashed = [
            (state, state.pop("use_muon"))
            for state in self.optimizer.state.values()
            if "use_muon" in state
        ]
        try:
            return super().sharded_state_dict(*args, **kwargs)
        finally:
            for state, value in stashed:
                state["use_muon"] = value


def get_megatron_muon_kimi_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
) -> Any:
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim._kimi_muon import Muon

    if config.use_distributed_optimizer:
        raise ValueError(
            "muon_kimi does not support the distributed optimizer (single-GPU dev only)."
        )
    if config.fp16:
        raise ValueError("muon_kimi does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("muon_kimi does not support tensor parallelism > 1 (single-GPU dev only).")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError(
            "muon_kimi does not support pipeline parallelism > 1 (single-GPU dev only)."
        )

    # Param split mirrors the native muon path (third_party/Megatron-LM/
    # megatron/core/optimizer/muon.py:283-302): 2-D non-embedding/output -> Muon,
    # everything else (embeddings, lm_head, norms, biases) -> internal AdamW.
    muon_params: list = []
    adamw_params: list = []
    for model_chunk in model_chunks:
        for _name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

    # Diagnostic: surface the routing split so the first run can confirm the
    # embedding / LM-head / norms landed in AdamW (not Muon). A 2-D embedding
    # with an unset is_embedding_or_output_parameter flag would silently route
    # to Muon and still pass Muon's ndim==2 assert — this log makes it visible.
    logger.info(
        "muon_kimi: %d muon params (2D non-embedding), %d adamw params",
        len(muon_params),
        len(adamw_params),
    )
    if not muon_params:
        logger.warning(
            "muon_kimi: no 2D non-embedding params found — Muon is a no-op (pure AdamW)."
        )

    # NOTE: weight decay is constant here. Muon reads group["wd"]; Megatron's
    # scheduler writes group["weight_decay"], which Muon ignores. This matches
    # the GaLore recipe (no WD schedule).
    optimizer = Muon(
        lr=config.lr,
        wd=config.weight_decay,
        muon_params=muon_params,
        momentum=config.muon_momentum,
        nesterov=config.muon_use_nesterov,
        ns_steps=config.muon_num_ns_steps,
        adamw_params=adamw_params,
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
    )

    def init_state_fn(opt, _config=None):
        # Called only during checkpoint (re)load; param_groups then hold the fp32
        # master params and opt.state[p]["use_muon"] has been transferred onto them
        # by Float16OptimizerWithFloat16Params.__init__. Fresh runs init lazily in
        # Muon.step(), so this fn does nothing there.
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                if state.get("use_muon", False):
                    state.setdefault("momentum_buffer", torch.zeros_like(p.data))
                else:
                    if "moment1" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(p.data)
                        state["moment2"] = torch.zeros_like(p.data)

    # Wrap with the mixin so the per-param `use_muon` bool is stripped while
    # Megatron builds the sharded checkpoint (otherwise the end-of-training save
    # crashes and the final-step validation is skipped — see the mixin docstring).
    if config.bf16:
        cls = type(
            "MuonKimiFloat16Optimizer",
            (_StripUseMuonShardingMixin, Float16OptimizerWithFloat16Params),
            {},
        )
        return cls(optimizer, config, None, init_state_fn)
    cls = type(
        "MuonKimiFP32Optimizer",
        (_StripUseMuonShardingMixin, FP32Optimizer),
        {},
    )
    return cls(optimizer, config, init_state_fn)
