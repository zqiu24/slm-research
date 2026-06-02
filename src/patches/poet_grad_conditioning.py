# src/patches/poet_grad_conditioning.py
"""Patch (Probe 0B): log per-block ∂f/∂Q conditioning during a POET run.

Env-gated by SLM_POET_GRAD_CONDITIONING=1 (interval via
SLM_POET_GRAD_CONDITIONING_INTERVAL, default 2000). Inert otherwise, so it is
safe in _ALWAYS_ON_PATCHES.

Mechanism: wrap ``setup_model_and_optimizer`` (NOT ``get_megatron_optimizer``).
The POET path goes through ``poet_optimizer_setup``, which — applied AFTER this
patch in sorted order — becomes the OUTER wrapper of ``get_megatron_optimizer``
and, for ``slm_optimizer=='poet'``, routes straight to
``get_megatron_poet_optimizer`` WITHOUT calling the original it wrapped. So a
wrapper on ``get_megatron_optimizer`` is dead on the POET path. Instead we wrap
``setup_model_and_optimizer`` (the same hook ``wandb_trainable_params`` uses),
which returns the fully-built ``(model, optimizer, scheduler)`` regardless of how
the optimizer was routed. We then (a) pick ~8 representative oft_R blocks from
``model`` and (b) wrap the ``optimizer``'s ``.step`` so that, every ``interval``
steps, it reads each block's gradient from ``main_grad`` (the Megatron DDP fp32
buffer; falls back to ``.grad``), reconstructs the skew, and logs spectral stats
to W&B BEFORE the optimizer consumes the grad.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

# Runtime wrapper with no static target ownership (like wandb_trainable_params):
# it monkeypatches setup_model_and_optimizer at apply() time and composes with any
# other wrapper of that symbol, so it registers targets=() to avoid a PatchConflict.
logger = logging.getLogger(__name__)

# projection name fragments we care about (HF-ish + Megatron names)
_WANTED = (
    "linear_q",
    "q_proj",
    "linear_v",
    "v_proj",
    "linear_fc2",
    "down_proj",
    "linear_fc1",
    "up_proj",
)


def select_target_params(named_layers, max_targets: int = 8):
    """From (name, poet_layer) pairs, pick representative blocks to probe.

    Returns a list of dicts: {label, factor ('R_in'|'R_out'), param, block_size}.
    A layer contributes a target only if its name matches a wanted projection.
    """
    targets = []
    for name, layer in named_layers:
        if not any(w in name for w in _WANTED):
            continue
        for factor, attr, bsz_attr in (
            ("R_in", "oft_R_in", "block_size_in"),
            ("R_out", "oft_R_out", "block_size_out"),
        ):
            param = getattr(layer, attr, None)
            bsz = getattr(layer, bsz_attr, None)
            if param is None or bsz is None:
                continue
            targets.append(
                {
                    "label": f"{name}.{factor}",
                    "factor": factor,
                    "param": param,
                    "block_size": int(bsz),
                    "layer": layer,
                }
            )
            if len(targets) >= max_targets:
                return targets
    return targets


def _log_conditioning(targets, iteration: int) -> None:
    import torch

    from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew

    try:
        import wandb
    except Exception:
        wandb = None

    for t in targets:
        param = t["param"]
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            logger.warning("[COND] no grad for %s at iter %d", t["label"], iteration)
            continue
        vec = grad.detach().to(torch.float32).reshape(param.shape[0], -1)
        skew = vec_to_skew(vec, t["block_size"])
        stats = block_spectral_stats(skew)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    f"poet_cond/{t['label']}/condition_number": stats["condition_number"]
                    .mean()
                    .item(),
                    f"poet_cond/{t['label']}/stable_rank": stats["stable_rank"].mean().item(),
                    f"poet_cond/{t['label']}/sigma_max_over_median": stats["sigma_max_over_median"]
                    .mean()
                    .item(),
                },
                step=iteration,
            )

        # rotation diagnostics: build R = f(Q) for this block's CURRENT oft_R and
        # log realized-angle ||G-I|| + orthogonality-drift ||RR^T-I|| (Stage-2 calibration).
        try:
            from poet_torch.poet_layer import pytorch_skew_symmetric

            from src.diag.rotation_diag import block_rotation_diagnostics

            oft = param.detach()
            bs = t["block_size"]
            rows, cols = torch.triu_indices(bs, bs, 1, device=oft.device)
            q_skew = pytorch_skew_symmetric(
                oft.float(), bs, rows.to(torch.int32), cols.to(torch.int32)
            )
            param_kind = getattr(t.get("layer"), "parameterization", "cayley")
            rot = (
                torch.linalg.matrix_exp(q_skew)
                if param_kind == "exp"
                else torch.ops.poet.cayley(q_skew)[0]
            )
            rd = block_rotation_diagnostics(rot)
            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(
                    {
                        f"poet_rot/{t['label']}/g_minus_i": rd["g_minus_i"].mean().item(),
                        f"poet_rot/{t['label']}/ortho_err": rd["ortho_err"].mean().item(),
                    },
                    step=iteration,
                )
        except Exception:  # diagnostics must never break training
            logger.exception("[COND] rotation diag failed for %s", t["label"])


def _install_step_hook(optimizer, targets, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        if state["n"] % interval == 0:
            try:
                _log_conditioning(targets, state["n"])
            except Exception:  # — diagnostics must never break training
                logger.exception("[COND] conditioning log failed at step %d", state["n"])
        state["n"] += 1
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


def _install_conditioning_on_setup(orig_setup, interval):
    """Wrap ``setup_model_and_optimizer`` to install the conditioning step-hook on
    the fully-built optimizer. Factored out so the wrap logic is unit-testable
    without a real Megatron import."""

    def _wrapped_setup(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig_setup(*args, **kwargs)
        chunks = model if isinstance(model, list) else [model]
        named_layers = [
            (name, mod)
            for m in chunks
            for name, mod in m.named_modules()
            if hasattr(mod, "oft_R_in") or hasattr(mod, "oft_R_out")
        ]
        targets = select_target_params(named_layers)
        if targets:
            _install_step_hook(optimizer, targets, interval)
            logger.warning(
                "[COND] ∂f/∂Q conditioning ENABLED — probing %d blocks every %d steps",
                len(targets),
                interval,
            )
        else:
            logger.warning("[COND] no oft_R layers found; conditioning probe is a no-op")
        return model, optimizer, opt_param_scheduler

    return _wrapped_setup


@register_patch(name="poet_grad_conditioning", targets=())
def apply() -> None:
    if os.environ.get("SLM_POET_GRAD_CONDITIONING") != "1":
        return  # inert unless explicitly enabled

    interval = int(os.environ.get("SLM_POET_GRAD_CONDITIONING_INTERVAL", "2000"))
    from megatron.training import training as _mt

    _mt.setup_model_and_optimizer = _install_conditioning_on_setup(
        _mt.setup_model_and_optimizer, interval
    )
