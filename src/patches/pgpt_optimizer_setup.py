"""Patch: pgpt optimizer-side setup — no-WD scaling groups + targeted renorm.

pgpt is POET-required, so it must NOT wrap ``get_megatron_optimizer`` (POET's
``poet_optimizer_setup`` owns that). Instead this patch cooperatively wraps
``megatron.training.training.setup_model_and_optimizer`` — the same hook
``wandb_trainable_params`` / ``poet_grad_conditioning`` use — and registers
``targets=()`` so it never raises a PatchConflict and composes regardless of
apply order.

After ``setup_model_and_optimizer`` returns it, when ``args.ngpt`` is set:
  (a) moves the nGPT scaling params (sqk/suv/attn_alpha/mlp_alpha/_ngpt_sz) into a
      zero-weight-decay group (belt-and-suspenders; the pgpt config also sets the
      global weight_decay to 0), and
  (b) installs a per-step L2 re-projection of the two sphere matrices POET does
      NOT wrap (token embedding + lm_head) by monkey-patching ``optimizer.step``.
      The per-layer POET-wrapped matrices are intentionally NOT re-projected —
      POET preserves their spectrum. The set comes from
      ``model._pgpt_post_step_norm_role_map`` (registered by pgpt_apply_spec).

NOTE (inherited from nGPT): the renorm mutates the *model* params via the role
map, exactly like nGPT's ``ngpt_normalize_step``. Under a float16 master-weight
optimizer the fp32 master copy is the source of truth; this behavior matches the
validated nGPT path and is confirmed by the GPU smoke, not introduced here.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)

# Trailing module-name segments holding an nGPT scaling vector as ``.param``.
_SCALING_MODULE_NAMES = frozenset({"sqk", "suv", "attn_alpha", "mlp_alpha", "_ngpt_sz"})


def classify_pgpt_scaling_params(model) -> list:
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] == "param" and parts[-2] in _SCALING_MODULE_NAMES:
            out.append(p)
    return out


def _install_renorm_step(optimizer, role_maps) -> None:
    """Monkey-patch ``optimizer.step`` to re-project embedding+lm_head after each step."""
    from src.model.pgpt.normalize import normalize_module_matrices

    if getattr(optimizer, "_pgpt_renorm_installed", False) or not role_maps:
        return
    orig_step = optimizer.step

    def _step(*a, **kw):
        ret = orig_step(*a, **kw)
        for role_map in role_maps:
            normalize_module_matrices(role_map)
        return ret

    optimizer.step = _step  # type: ignore[assignment]
    optimizer._pgpt_renorm_installed = True


@register_patch(name="pgpt_optimizer_setup", targets=())
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    orig = _mt.setup_model_and_optimizer
    if getattr(orig, "_pgpt_optimizer_setup", False):
        return

    def _wrapped(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig(*args, **kwargs)
        if not getattr(get_args(), "ngpt", False):
            return model, optimizer, opt_param_scheduler

        chunks = model if isinstance(model, list | tuple) else [model]

        # (a) zero-WD for scaling params
        scaling_ids: set[int] = set()
        for m in chunks:
            scaling_ids.update(id(p) for p in classify_pgpt_scaling_params(m))
        inner = getattr(optimizer, "optimizer", None) or optimizer
        for group in getattr(inner, "param_groups", []):
            if any(id(p) in scaling_ids for p in group["params"]):
                group["weight_decay"] = 0.0

        # (b) targeted per-step renorm of embedding + lm_head
        role_maps = [
            rm for rm in (getattr(m, "_pgpt_post_step_norm_role_map", None) for m in chunks) if rm
        ]
        _install_renorm_step(optimizer, role_maps)

        logger.info(
            "[pgpt] optimizer setup: zero-WD scaling groups + embedding/lm_head renorm hook"
        )
        return model, optimizer, opt_param_scheduler

    _wrapped._pgpt_optimizer_setup = True
    _mt.setup_model_and_optimizer = _wrapped
