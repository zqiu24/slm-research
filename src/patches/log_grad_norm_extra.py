"""Patch: log the post-clip ("actual") gradient norm alongside the raw one.

Megatron logs ``grad-norm`` = the RAW total-norm, computed *before* clipping
(``get_grad_norm_fp32`` returns it, then ``clip_grad_by_total_norm_fp32`` scales
the grads). This patch wraps ``training_log`` and adds two scalars:

* ``grad-norm-clipped``    = min(raw_norm, clip_grad) — the norm of the gradient
  actually fed to the optimizer. NOTE: while the raw norm exceeds ``clip_grad``
  this sits flat at ``clip_grad`` (e.g. 1.0); it only tracks the raw norm once
  the raw norm drops below the clip threshold.
* ``grad-norm-clip-coeff`` = min(1, clip_grad/raw_norm) — the scale factor applied
  to the gradient (1.0 = not clipped, ~1e-5 = clipped hard). The more dynamic of
  the two: it rises toward 1.0 as the raw norm falls.

Targets ``megatron.training.training.training_log`` (no other patch touches it).
"""

from __future__ import annotations

import contextlib

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.training_log",)


@register_patch(name="log_grad_norm_extra", targets=_TARGET)
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    _orig = _mt.training_log
    if getattr(_orig, "_slm_grad_extra_wrapped", False):
        return

    def _wrapped(*a, **kw):
        ret = _orig(*a, **kw)
        # Logging must never break training.
        with contextlib.suppress(Exception):
            # training_log(loss_dict, total_loss_dict, learning_rate, iteration,
            #   loss_scale, report_memory_flag, skipped_iter, grad_norm, ...)
            grad_norm = kw["grad_norm"] if "grad_norm" in kw else (a[7] if len(a) > 7 else None)
            iteration = kw["iteration"] if "iteration" in kw else (a[3] if len(a) > 3 else None)
            if grad_norm is not None and iteration is not None:
                args = get_args()
                writer = _mt.get_tensorboard_writer()
                wandb_writer = _mt.get_wandb_writer()
                if writer and iteration % args.tensorboard_log_interval == 0:
                    clip = float(getattr(args, "clip_grad", 0.0) or 0.0)
                    applied = min(grad_norm, clip) if clip > 0 else grad_norm
                    coeff = min(1.0, clip / (grad_norm + 1.0e-6)) if clip > 0 else 1.0
                    writer.add_scalar("grad-norm-clipped", applied, iteration)
                    writer.add_scalar(
                        "grad-norm-clipped vs samples", applied, args.consumed_train_samples
                    )
                    writer.add_scalar("grad-norm-clip-coeff", coeff, iteration)
                    if wandb_writer:
                        wandb_writer.log(
                            {"grad-norm-clipped": applied, "grad-norm-clip-coeff": coeff},
                            iteration,
                        )
        return ret

    _wrapped._slm_grad_extra_wrapped = True
    _mt.training_log = _wrapped
