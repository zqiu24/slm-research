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
        # TEMP VERIFY (POET_VERIFY=1; delete later): confirm POETAdam.step is
        # actually invoked (Q2 — it is reached via Float16Optimizer.step_with_ready_grads).
        if os.environ.get("POET_VERIFY", "0") == "1" and self.global_step_counter <= 3:
            print(
                f"[POET-VERIFY] POETAdam.step CALLED (counter={self.global_step_counter}, "
                f"poet_merge_period={self.poet_merge_period})",
                flush=True,
            )
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
        # TEMP VERIFY (POET_VERIFY=1; delete later): track that the merge-period
        # reset actually zeroes the Adam step counter for oft_R params (Q2).
        _verify = os.environ.get("POET_VERIFY", "0") == "1"
        _n_step = 0
        _sample_before = None
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                st = self.base_optimizer.state.get(p, {})
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].zero_()
                if "step" in st:
                    if _verify and _sample_before is None:
                        _s = st["step"]
                        _sample_before = (
                            float(_s.item()) if isinstance(_s, torch.Tensor) else float(_s)
                        )
                    if isinstance(st["step"], torch.Tensor):
                        st["step"].zero_()
                    else:
                        st["step"] = 0
                    _n_step += 1
        # Some Adam impls (apex/fused) store `step` per param-group rather than
        # per-param, so the per-param reset above is a no-op for them. Reset the
        # group step too — every base_optimizer group here holds only oft_R
        # params, so this is safe. t -> 0 gives fresh bias correction post-merge.
        _grp_before = None
        _n_groups = 0
        for group in self.base_optimizer.param_groups:
            if "step" in group:
                if _verify and _grp_before is None:
                    _gs = group["step"]
                    _grp_before = float(_gs.item()) if isinstance(_gs, torch.Tensor) else float(_gs)
                if isinstance(group["step"], torch.Tensor):
                    group["step"].zero_()
                else:
                    group["step"] = 0
                _n_groups += 1
        if _verify:
            print(
                f"[POET-VERIFY] _reset_momentum FIRED (counter={self.global_step_counter}): "
                f"zeroed momentum for {_n_step} oft_R params; per-param step "
                f"{_sample_before} -> 0; group-level step {_grp_before} -> 0 "
                f"({_n_groups} groups)",
                flush=True,
            )

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
    """Stock Megatron optimizer — the DEFAULT POET path (``use_poet_adam=false``).

    No POETAdam, no manual linear/nonlinear partition, no ``ChainedOptimizer``
    plumbing: POETLinear already froze the base weights, and Megatron's
    ``_get_param_groups`` skips ``requires_grad=False`` params, so the stock
    builder optimizes exactly ``oft_R + embeddings + norms`` — the same trainable
    set the custom POETAdam path produces.

    ``poet_scale`` is applied to ``oft_R`` only, via a per-parameter
    ``max_lr``/``min_lr`` override keyed on the ``*oft_R*`` name glob (the
    scheduler honours per-group ``max_lr``/``min_lr``; ``lr_mult`` would only
    scale weight-decay). This mirrors the LR scaling POETAdam applied at
    construction. The periodic Adam momentum reset is performed by the
    ``poet_merge_step`` hook (which skips it only when ``use_poet_adam=true``,
    since POETAdam resets momentum itself).
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
    opt = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    _verify_poet_groups(opt, model_chunks)  # TEMP VERIFY (POET_VERIFY=1; delete later)
    return opt


def _verify_poet_groups(chained, model_chunks) -> None:
    """TEMP VERIFY (POET_VERIFY=1; delete later). Dump each optimizer param
    group's lr / max_lr / weight_decay, labeling oft_R groups vs the rest, so a
    dev run can confirm:
      Q1 — oft_R (POET) groups have a different LR than normal params (poet_scale).
      Q3 — which groups get weight decay.
    """
    if os.environ.get("POET_VERIFY", "0") != "1":
        return
    try:
        rank = (
            torch.distributed.get_rank()
            if (torch.distributed.is_available() and torch.distributed.is_initialized())
            else 0
        )
        if rank != 0:
            return
        id2name = {}
        for mc in model_chunks:
            for n, p in mc.named_parameters():
                id2name[id(p)] = n
        print(
            "[POET-VERIFY] ===== optimizer param groups (lr / max_lr / weight_decay) =====",
            flush=True,
        )
        opt_list = getattr(chained, "chained_optimizers", None) or [chained]
        for oi, opt in enumerate(opt_list):
            inner = getattr(opt, "optimizer", opt)
            # optimizer param_groups hold fp32 MASTER params; map them back to
            # the model params so id2name can resolve oft_R vs the rest.
            master2model = {}
            f16 = getattr(opt, "float16_groups", None)
            fp32m = getattr(opt, "fp32_from_float16_groups", None)
            if f16 and fp32m:
                for fg, mg in zip(f16, fp32m, strict=False):
                    for model_p, master_p in zip(fg, mg, strict=False):
                        master2model[id(master_p)] = model_p
            for gi, g in enumerate(getattr(inner, "param_groups", [])):
                ps = g.get("params", [])
                names = [id2name.get(id(master2model.get(id(p), p)), "?") for p in ps]
                n_oft = sum(1 for nm in names if "oft_R" in nm)
                kind = "OFT" if (ps and n_oft == len(ps)) else ("MIXED" if n_oft else "non-oft")
                sample = next((nm for nm in names if nm != "?"), names[0] if names else "")
                shape0 = tuple(ps[0].shape) if ps else None
                print(
                    f"[POET-VERIFY] chain[{oi}].group[{gi}] kind={kind} "
                    f"lr={g.get('lr')} max_lr={g.get('max_lr')} min_lr={g.get('min_lr')} "
                    f"weight_decay={g.get('weight_decay')} wd_mult={g.get('wd_mult')} "
                    f"nparams={len(ps)} n_oft={n_oft} shape0={shape0} sample={sample}",
                    flush=True,
                )
        print("[POET-VERIFY] ===== end param groups =====", flush=True)
    except Exception as e:
        print(f"[POET-VERIFY] group dump failed: {e}", flush=True)


def _split_poet_muon_params(model_chunks):
    """oft_R params -> skew (SkewMuon); all other trainable params -> AdamW."""
    skew_params, adamw_params = [], []
    for mc in model_chunks:
        for name, param in mc.named_parameters():
            if not param.requires_grad:
                continue
            (skew_params if "oft_R" in name else adamw_params).append(param)
    return skew_params, adamw_params


def get_megatron_poet_muon_optimizer(
    config,
    model_chunks,
    *,
    config_overrides=None,
    use_gloo_process_groups: bool = True,
):
    """POET Muon-on-Q: SkewMuon on oft_R, AdamW on the rest, wrapped for Megatron.
    Single-process / DP-replicated (no sharded distributed optimizer), like
    muon_kimi. Designed for the no-reset regime (merge_period=0)."""
    _resolve_megatron_handles()
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim.poet_skew_muon import SkewMuon

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("POET Muon-on-Q does not support the distributed optimizer (dev only).")
    if getattr(config, "fp16", False):
        raise ValueError("POET Muon-on-Q does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("POET Muon-on-Q does not support tensor parallelism > 1.")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("POET Muon-on-Q does not support pipeline parallelism > 1.")

    skew_params, adamw_params = _split_poet_muon_params(model_chunks)
    logger.info(
        "[POET] Muon-on-Q: %d skew (oft_R) params, %d adamw params (theta=%s, ns_steps=%s)",
        len(skew_params),
        len(adamw_params),
        getattr(config, "poet_muon_theta", 0.1),
        getattr(config, "poet_muon_ns_steps", 5),
    )
    if not skew_params:
        logger.warning("[POET] Muon-on-Q: no oft_R params found — SkewMuon is a no-op.")

    optimizer = SkewMuon(
        skew_params=skew_params,
        adamw_params=adamw_params,
        theta=getattr(config, "poet_muon_theta", 0.1),
        ns_steps=getattr(config, "poet_muon_ns_steps", 5),
        momentum=getattr(config, "poet_muon_momentum", 0.95),
        nesterov=True,
        adamw_lr=config.lr,
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )

    def init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state[p]
                if st.get("use_skew", False):
                    st.setdefault("momentum_buffer", torch.zeros_like(p.data))
                elif "moment1" not in st:
                    st["step"] = 0
                    st["moment1"] = torch.zeros_like(p.data)
                    st["moment2"] = torch.zeros_like(p.data)

    if getattr(config, "bf16", False):
        return Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
    return FP32Optimizer(optimizer, config, init_state_fn)


def get_megatron_poet_lie_momentum_optimizer(
    config,
    model_chunks,
    *,
    config_overrides=None,
    use_gloo_process_groups: bool = True,
):
    """POET Lie-algebra momentum: LieAlgebraMomentum on oft_R, AdamW on the rest,
    wrapped for Megatron. Single-process / DP-replicated (no sharded distributed
    optimizer), like the muon path. Increment 1 of POET-X x Pion."""
    _resolve_megatron_handles()
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim.poet_lie_momentum import (
        LieAlgebraMomentum,
        _build_lie_param_groups,
        _split_poet_lie_params,
    )

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("POET Lie-momentum does not support the distributed optimizer (dev only).")
    if getattr(config, "fp16", False):
        raise ValueError("POET Lie-momentum does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("POET Lie-momentum does not support tensor parallelism > 1.")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("POET Lie-momentum does not support pipeline parallelism > 1.")

    skew_in, skew_out, adamw_params = _split_poet_lie_params(model_chunks)
    scale = getattr(config, "poet_scale", 1.0)
    min_lr = getattr(config, "min_lr", 0.0)
    logger.info(
        "[POET] Lie-momentum: %d in + %d out skew (oft_R) params, %d adamw "
        "(b1=%s, b2=%s, v_mode=%s, scale=%s, alternating=%s, alternate_every=%s)",
        len(skew_in),
        len(skew_out),
        len(adamw_params),
        getattr(config, "poet_lie_b1", 0.9),
        getattr(config, "poet_lie_b2", 0.95),
        getattr(config, "poet_lie_v_mode", "elementwise"),
        scale,
        getattr(config, "poet_lie_alternating", False),
        getattr(config, "poet_lie_alternate_every", 1),
    )
    if not (skew_in or skew_out):
        logger.warning("[POET] Lie-momentum: no oft_R params found — skew branch is a no-op.")

    param_groups = _build_lie_param_groups(
        skew_in, skew_out, adamw_params, config.lr, min_lr, scale
    )
    optimizer = LieAlgebraMomentum(
        param_groups,
        b1=getattr(config, "poet_lie_b1", 0.9),
        b2=getattr(config, "poet_lie_b2", 0.95),
        eps=getattr(config, "poet_lie_eps", 1e-8),
        v_mode=getattr(config, "poet_lie_v_mode", "elementwise"),
        alternating=getattr(config, "poet_lie_alternating", False),
        alternate_every=getattr(config, "poet_lie_alternate_every", 1),
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )

    def init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state[p]
                if group["use_skew"]:
                    st.setdefault("lie_m", torch.zeros_like(p.data))
                    if group["v_mode"] == "scalar":
                        st.setdefault(
                            "lie_v",
                            torch.zeros(
                                p.data.shape[0], 1, dtype=p.data.dtype, device=p.data.device
                            ),
                        )
                    else:
                        st.setdefault("lie_v", torch.zeros_like(p.data))
                elif "moment1" not in st:
                    st["step"] = 0
                    st["moment1"] = torch.zeros_like(p.data)
                    st["moment2"] = torch.zeros_like(p.data)

    if getattr(config, "bf16", False):
        return Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
    return FP32Optimizer(optimizer, config, init_state_fn)


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

    if getattr(config, "poet_q_optimizer", "adam") == "lie_algebra":
        return get_megatron_poet_lie_momentum_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )

    if getattr(config, "poet_q_optimizer", "adam") == "muon":
        return get_megatron_poet_muon_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )

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

    # Optimizer-impl selection. DEFAULT = the Megatron-Adam path: the stock
    # Megatron optimizer, with oft_R's LR scaled via a param-group override and
    # its Adam momentum reset by the poet_merge_step hook. The custom POETAdam +
    # ChainedOptimizer path is opt-in via ``optim.poet.use_poet_adam=true``.
    use_poet_adam = bool(getattr(config, "poet_use_poet_adam", False))
    if not use_poet_adam:
        if poet_cache_mode != "none":
            raise ValueError(
                "The Megatron-Adam POET path supports only poet_cache_mode='none'; "
                "the cached_fwd_bwd path needs the POETAdam VJP flush hook "
                f"(got {poet_cache_mode!r}). Set optim.poet.use_poet_adam=true to use it."
            )
        logger.info(
            "[POET] Megatron-Adam path (no POETAdam): oft_R LR x%s via param-group "
            "override; Adam momentum reset via poet_merge_step hook.",
            poet_scale,
        )
        return _build_vanilla_poet_optimizer(
            config,
            model_chunks,
            poet_scale,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )

    logger.info("[POET] custom POETAdam + ChainedOptimizer path (optim.poet.use_poet_adam=true).")

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

    chained = ChainedOptimizer([poet_wrapped, *chained_adam.chained_optimizers])
    _verify_poet_groups(chained, model_chunks)  # TEMP VERIFY (POET_VERIFY=1; delete later)
    return chained
