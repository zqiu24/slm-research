"""POET optimizer: Adam wrapper with periodic momentum reset + LR scaling.

Ported from fork 2 (/lustre/scratch/zqiu/Megatron-LM/megatron/core/optimizer/poet.py,
commit bb43fa063). The wrapper preserves the base optimizer's state dict and
``step()``, and adds:

* ``poet_merge_period`` — every N steps, zero ``exp_avg`` / ``exp_avg_sq`` /
  step counter on every param.
* ``poet_scale`` — multiplicative LR factor applied at construction (and
  also propagated into ``max_lr`` / ``min_lr`` so Megatron's LR scheduler
  warmup + cosine decay stay scale-consistent).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from src.optim import poet_cache as _pc

logger = logging.getLogger(__name__)


class POETAdam(torch.optim.Optimizer):
    """Wraps any Adam-like optimizer and adds POET momentum reset + LR scale."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        poet_merge_period: int = 0,
        poet_scale: float = 1.0,
        poet_cache_mode: str = "none",
    ):
        # We do NOT call super().__init__: state and param_groups are owned
        # by ``base_optimizer`` and we only proxy them.
        self.base_optimizer = base_optimizer
        self.poet_merge_period = poet_merge_period
        self.poet_scale = poet_scale
        self.global_step_counter = 0

        if poet_scale != 1.0:
            for group in self.base_optimizer.param_groups:
                group["lr"] = group["lr"] * poet_scale
                if "max_lr" not in group:
                    group["max_lr"] = group["lr"]
                else:
                    group["max_lr"] = group["max_lr"] * poet_scale
                if "min_lr" in group:
                    group["min_lr"] = group["min_lr"] * poet_scale
        else:
            # Even when not scaling, set max_lr so Megatron's LR scheduler
            # has something to clamp against. Match the upstream fork's
            # behaviour: only set if missing.
            for group in self.base_optimizer.param_groups:
                group.setdefault("max_lr", group["lr"])

        self.poet_cache_mode = poet_cache_mode
        _pc.set_cache_mode(poet_cache_mode)

    # ---- proxy attributes -------------------------------------------------

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def state_dict(self):
        sd = self.base_optimizer.state_dict()
        sd["poet_global_step_counter"] = self.global_step_counter
        return sd

    def load_state_dict(self, state_dict):
        self.global_step_counter = state_dict.pop("poet_global_step_counter", 0)
        # Spec §11: caches built against pre-load oft_R are stale.
        _pc.bump_poet_version()
        _pc.invalidate_all_poet_caches()
        self.base_optimizer.load_state_dict(state_dict)

    def zero_grad(self, *args, **kwargs):
        return self.base_optimizer.zero_grad(*args, **kwargs)

    # ---- step + periodic reset -------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        ret = self.base_optimizer.step(closure)
        self.global_step_counter += 1
        if _pc.get_cache_mode() != "none":
            _pc.bump_poet_version()
        if self.poet_merge_period > 0 and self.global_step_counter % self.poet_merge_period == 0:
            logger.info(
                "POET: resetting Adam momentum at global step %d",
                self.global_step_counter,
            )
            self._reset_momentum()
        return ret

    def _reset_momentum(self) -> None:
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                st = self.base_optimizer.state.get(p, {})
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].zero_()
                if "step" in st:
                    if isinstance(st["step"], torch.Tensor):
                        st["step"].zero_()
                    else:
                        st["step"] = 0

    # Forward any other attribute access to the base optimizer.
    def __getattr__(self, name: str) -> Any:
        if name in (
            "base_optimizer",
            "poet_merge_period",
            "poet_scale",
            "global_step_counter",
        ):
            raise AttributeError(name)
        return getattr(self.base_optimizer, name)


# --------------------------------------------------------------------------
# Megatron-aware builder
# --------------------------------------------------------------------------

# Lazy module-level handles; real refs are populated by
# ``_resolve_megatron_handles`` on first call. Tests monkeypatch these.
_get_param_groups = None
get_megatron_optimizer = None
ChainedOptimizer = None
Float16OptimizerWithFloat16Params = None
FP32Optimizer = None
_BaseAdamCls = None
_USING_PYTORCH_OPTIMIZER = None


def _resolve_megatron_handles() -> None:
    """Import Megatron optimizer primitives on first use.

    Done lazily so unit tests that don't need them (and CPU-only environments)
    can still import this module.
    """
    global _get_param_groups, get_megatron_optimizer, ChainedOptimizer
    global Float16OptimizerWithFloat16Params, FP32Optimizer
    global _BaseAdamCls, _USING_PYTORCH_OPTIMIZER
    if _get_param_groups is not None:
        return

    from megatron.core.optimizer import (
        USING_PYTORCH_OPTIMIZER as _UPO,
    )
    from megatron.core.optimizer import (
        Adam as _Adam,
    )
    from megatron.core.optimizer import (
        _get_param_groups as _gpg,
    )
    from megatron.core.optimizer import (
        get_megatron_optimizer as _gmo,
    )
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer as _ChainedOptimizer,
    )
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params as _Float16OptimizerWithFloat16Params,
    )
    from megatron.core.optimizer.optimizer import (
        FP32Optimizer as _FP32Optimizer,
    )

    _get_param_groups = _gpg
    get_megatron_optimizer = _gmo
    ChainedOptimizer = _ChainedOptimizer
    Float16OptimizerWithFloat16Params = _Float16OptimizerWithFloat16Params
    FP32Optimizer = _FP32Optimizer
    _BaseAdamCls = _Adam
    _USING_PYTORCH_OPTIMIZER = _UPO


def _is_distributed_dp() -> bool:
    """True when a DP process group of size > 1 is initialized."""
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
    except Exception:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return mpu.get_data_parallel_world_size() > 1
    except Exception:
        return False


def _sync_oft_R_grads_across_dp(layers) -> None:  # noqa: N802
    """All-reduce oft_R.main_grad across the DP group.

    Mode A populates oft_R.main_grad via _flush_R_grads_to_oft_R, AFTER
    Megatron's DDP grad reducer has already finished its work for this
    backward. The reducer never saw our update, so we sync explicitly.

    Packs every layer's main_grad into one flat buffer for a single
    allreduce rather than one per layer.

    Safe no-op outside a real DP world (CPU dev box, single-rank GPU).
    """
    if not _is_distributed_dp():
        return
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    grads = []
    for layer in layers:
        # Decoupled layers expose oft_R_in / oft_R_out; legacy callers may
        # still expose a single oft_R. Sync whichever exist.
        for name in ("oft_R_in", "oft_R_out", "oft_R"):
            p = getattr(layer, name, None)
            if p is None:
                continue
            if hasattr(p, "main_grad") and p.main_grad is not None:
                grads.append(p.main_grad)
            elif p.grad is not None:
                grads.append(p.grad)
    if not grads:
        return
    dp_group = mpu.get_data_parallel_group()
    ws = mpu.get_data_parallel_world_size()
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, group=dp_group)
    flat.div_(ws)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=False):
        g.copy_(synced)


def _flush_poet_caches_for_step() -> None:
    """Walk live POET layers, flush each one's R-leaf grads into
    oft_R.main_grad (or .grad fallback), then all-reduce across DP."""
    with torch.enable_grad():
        layers = list(_pc.iter_live_layers())
        for layer in layers:
            layer._flush_R_grads_to_oft_R()
    _sync_oft_R_grads_across_dp(layers)


def _install_poet_step_hook(wrapped_optimizer, cache_mode: str) -> None:
    """Install a pre-flush hook on the outer optimizer wrapper's prepare_grads.

    ``get_megatron_poet_optimizer`` returns a ``ChainedOptimizer``.
    ``ChainedOptimizer.step()`` (Megatron source, line 1317) drives each
    child by calling ``child.prepare_grads()`` then
    ``child.step_with_ready_grads()`` — it NEVER calls ``child.step()``.
    ``prepare_grads()`` is what invokes ``_copy_model_grads_to_main_grads``,
    which copies ``param.main_grad → main_param.grad``.

    For Mode A to work, the flush must write ``oft_R.main_grad`` (and
    all-reduce across DP) BEFORE ``_copy_model_grads_to_main_grads`` runs.
    Wrapping ``wrapped_optimizer.prepare_grads`` at the instance level
    achieves this: the hook runs first, then the original ``prepare_grads``
    does its normal work — including copying main_grad into main_param.grad.

    ``prepare_grads`` returns a ``found_inf_flag`` bool that
    ``ChainedOptimizer.prepare_grads`` ORs across all children; we MUST
    return the original's return value unchanged.

    Only ``cache_mode == "cached_fwd_bwd"`` needs this hook.
    """
    if cache_mode != "cached_fwd_bwd":
        return
    orig_prepare_grads = wrapped_optimizer.prepare_grads

    def _wrapped_prepare_grads(*a, **kw):
        # Mode A's per-microbatch backward writes only to detached R-leaves,
        # so oft_R.main_grad is still zero here. Flush our manual VJP into
        # oft_R.main_grad (+ DP all-reduce) BEFORE the original prepare_grads
        # copies main_grad -> main_param.grad via _copy_model_grads_to_main_grads.
        _flush_poet_caches_for_step()
        return orig_prepare_grads(*a, **kw)

    wrapped_optimizer.prepare_grads = _wrapped_prepare_grads


def _build_vanilla_poet_optimizer(
    config: Any,
    model_chunks: list,
    poet_scale: float,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
):
    """Stock Megatron optimizer for the ``POET_VANILLA_OPT`` A/B path.

    No POETAdam, no manual linear/nonlinear partition, no ``ChainedOptimizer``
    plumbing: POETLinear already froze the base weights, and Megatron's
    ``_get_param_groups`` skips ``requires_grad=False`` params, so the stock
    builder optimizes exactly ``oft_R + embeddings + norms`` — the same trainable
    set the custom path produces.

    ``poet_scale`` is applied to ``oft_R`` only, via a per-parameter
    ``max_lr``/``min_lr`` override keyed on the ``*oft_R*`` name glob (the
    scheduler honours per-group ``max_lr``/``min_lr``; ``lr_mult`` would only
    scale weight-decay). This mirrors the LR scaling POETAdam applied at
    construction. The periodic Adam momentum reset is reinstated by the
    ``poet_merge_step`` hook when ``POET_VANILLA_OPT`` is set.
    """
    from megatron.core.optimizer import get_standard_config_overrides
    from megatron.core.optimizer.optimizer_config import ParamKey

    overrides = (
        dict(config_overrides)
        if config_overrides is not None
        else dict(get_standard_config_overrides(config))
    )
    if poet_scale != 1.0:
        overrides[ParamKey(name="*oft_R*")] = {
            "max_lr": config.lr * poet_scale,
            "min_lr": config.min_lr * poet_scale,
        }
    return get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )


def get_megatron_poet_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
):
    """Build a ChainedOptimizer with POETAdam for linear-2D-non-embedding params
    and a regular Megatron optimizer for everything else.

    Mirrors fork 2's get_megatron_poet_optimizer (commit bb43fa063), rerouted
    through ``src.optim`` and slm-research's experiment-YAML schema.
    """
    _resolve_megatron_handles()

    poet_merge_period = getattr(config, "poet_merge_period", 0)
    poet_scale = getattr(config, "poet_scale", 1.0)
    poet_cache_mode = getattr(config, "poet_cache_mode", "none")

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("POET optimizer does not support distributed optimizer.")
    if getattr(config, "fp16", False):
        raise ValueError("POET optimizer does not support fp16.")

    # Force underlying optimizer family to adam so the chained-Adam path
    # for nonlinear params doesn't try to recurse into POET.
    if hasattr(config, "optimizer"):
        config.optimizer = "adam"

    logger.info(
        "Setting up POET optimizer: merge_period=%s, scale=%s",
        poet_merge_period,
        poet_scale,
    )

    # A/B path (POET_VANILLA_OPT=1): skip the custom POETAdam + manual partition
    # and use the STOCK Megatron optimizer. Lets us isolate whether POETAdam
    # itself contributes anything beyond stock Adam + a counted momentum reset.
    if os.environ.get("POET_VANILLA_OPT") == "1":
        if poet_cache_mode != "none":
            raise ValueError(
                "POET_VANILLA_OPT supports only poet_cache_mode='none'; the "
                "cached_fwd_bwd path needs the manual VJP flush hook that the "
                f"stock optimizer bypasses (got {poet_cache_mode!r})."
            )
        logger.warning(
            "[POET] POET_VANILLA_OPT=1 -> stock Megatron optimizer (no POETAdam); "
            "oft_R LR x%s via param-group override; reset via poet_merge_step hook.",
            poet_scale,
        )
        return _build_vanilla_poet_optimizer(
            config,
            model_chunks,
            poet_scale,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )

    def poet_init_state_fn(opt, config=None):
        base = opt.base_optimizer if hasattr(opt, "base_optimizer") else opt
        for group in base.param_groups:
            for p in group["params"]:
                if len(base.state[p]) == 0:
                    if config is None or not getattr(
                        config, "use_precision_aware_optimizer", False
                    ):
                        base.state[p]["exp_avg"] = torch.zeros_like(p.data)
                        base.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        base.initialize_state(p)

    # Partition: linear 2D non-embedding vs rest.
    linear_params: list[torch.nn.Parameter] = []
    nonlinear_params: list[torch.nn.Parameter] = []
    for mc in model_chunks:
        for _, param in mc.named_parameters():
            if not param.requires_grad:
                continue
            is_embed = getattr(param, "is_embedding_or_output_parameter", False)
            if not is_embed and param.dim() == 2:
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # Freeze nonlinear, build linear-only param groups for the POET branch.
    for p in nonlinear_params:
        p.requires_grad = False
    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    kwargs = dict(
        params=linear_param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    if _USING_PYTORCH_OPTIMIZER:
        adam_cls = (
            torch.optim.AdamW
            if getattr(config, "decoupled_weight_decay", False)
            else torch.optim.Adam
        )
    else:
        kwargs["adam_w_mode"] = getattr(config, "decoupled_weight_decay", False)
        adam_cls = _BaseAdamCls

    base_adam = adam_cls(**kwargs)
    poet_opt = POETAdam(
        base_adam,
        poet_merge_period=poet_merge_period,
        poet_scale=poet_scale,
        poet_cache_mode=poet_cache_mode,
    )

    if getattr(config, "bf16", False):
        poet_wrapped = Float16OptimizerWithFloat16Params(poet_opt, config, None, poet_init_state_fn)
    else:
        poet_wrapped = FP32Optimizer(poet_opt, config, poet_init_state_fn)
    _install_poet_step_hook(poet_wrapped, cache_mode=poet_cache_mode)

    # Now build the chained-Adam path for the nonlinear remainder.
    for p in nonlinear_params:
        p.requires_grad = True
    for p in linear_params:
        p.requires_grad = False
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    for p in linear_params:
        p.requires_grad = True

    return ChainedOptimizer([poet_wrapped, *chained_adam.chained_optimizers])
