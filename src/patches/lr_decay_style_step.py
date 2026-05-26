"""Patch: add ``step`` decay style to Megatron's ``OptimizerParamScheduler``.

Ports the piecewise-constant step-decay schedule from
``Megatron-poet/megatron/core/optimizer_param_scheduler.py``. After warmup
the LR coefficient stays at 1.0 until ``num_steps / lr_decay_steps`` reaches
the first ratio, then drops to the matching coefficient at each subsequent
ratio boundary. Past ``lr_decay_steps`` the LR is pinned at ``min_lr``
(upstream convention).

Ratio / coeff lists are read off Megatron's ``get_args()`` via two new CLI
flags wired through ``launchers/pretrain_gpt_slm.add_slm_args``:

  --lr-decay-step-ratio 0.8 0.9
  --lr-decay-step-coeff 0.316 0.1

(matches the Megatron-poet DeepSeek-3B training script defaults).
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGETS = (
    "megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.__init__",
    "megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.get_lr",
)


def _validate(ratio: list[float], coeff: list[float]) -> None:
    if ratio is None or coeff is None:
        raise ValueError(
            "--lr-decay-style=step requires both --lr-decay-step-ratio and --lr-decay-step-coeff"
        )
    if len(ratio) != len(coeff):
        raise ValueError(
            f"--lr-decay-step-ratio (len {len(ratio)}) and --lr-decay-step-coeff "
            f"(len {len(coeff)}) must have the same length"
        )
    if not all(0.0 < r < 1.0 for r in ratio):
        raise ValueError(f"--lr-decay-step-ratio values must be in (0, 1), got {ratio}")
    if list(ratio) != sorted(ratio):
        raise ValueError(f"--lr-decay-step-ratio must be sorted ascending, got {ratio}")


@register_patch(name="lr_decay_style_step", targets=_TARGETS)
def apply() -> None:
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    _orig_init = OptimizerParamScheduler.__init__
    _orig_get_lr = OptimizerParamScheduler.get_lr

    def patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
        _orig_init(self, *args, **kwargs)
        # Default: no step-decay knobs unless the user opted in.
        self.lr_decay_step_ratio = None
        self.lr_decay_step_coeff = None
        if getattr(self, "lr_decay_style", None) != "step":
            return
        try:
            from megatron.training.global_vars import get_args

            mg_args = get_args()
        except (AssertionError, ImportError) as exc:
            raise RuntimeError(
                "lr_decay_style='step' requires Megatron's get_args() to be initialized "
                "before OptimizerParamScheduler is constructed"
            ) from exc
        ratio = getattr(mg_args, "lr_decay_step_ratio", None)
        coeff = getattr(mg_args, "lr_decay_step_coeff", None)
        _validate(ratio, coeff)
        self.lr_decay_step_ratio = list(ratio)
        self.lr_decay_step_coeff = list(coeff)

    def patched_get_lr(self, param_group):  # type: ignore[no-redef]
        if getattr(self, "lr_decay_style", None) != "step":
            return _orig_get_lr(self, param_group)

        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
            )
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        progress = float(self.num_steps) / float(self.lr_decay_steps)
        coeff = 1.0
        for r, c in zip(self.lr_decay_step_ratio, self.lr_decay_step_coeff, strict=False):
            if progress >= r:
                coeff = c
        return min_lr + coeff * (max_lr - min_lr)

    OptimizerParamScheduler.__init__ = patched_init
    OptimizerParamScheduler.get_lr = patched_get_lr
