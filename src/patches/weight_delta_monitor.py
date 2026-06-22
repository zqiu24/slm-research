# src/patches/weight_delta_monitor.py
"""Patch: log materialized weight displacement ``delta_W`` to W&B.

Flag-gated by ``--log-delta-w`` (interval ``--log-delta-w-interval``, default
250; layers ``--delta-w-layers``, default ``first,mid,last``). Inert otherwise,
so it is safe in ``_ALWAYS_ON_PATCHES``.

Mechanism: wrap ``train_step`` as an outer wrapper. On a logging step, snapshot
selected 2-D weights before calling the inner ``train_step``, snapshot them again
after the inner call returns, and log metrics on ``W_after - W_before``. Because
``weight_delta_monitor`` sorts after ``poet_merge_step``, the after-snapshot in a
POET run sees the post-merge materialized weight on merge-boundary steps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from src.diag.skew_conditioning import block_spectral_stats
from src.patches._registry import register_patch
from src.patches.weight_norm_monitor import classify_linear, parse_layer_selection, should_log

logger = logging.getLogger(__name__)

_state = {
    "warned_merge0": False,
    "warned_cadence": False,
    "warned_tp": False,
    "warned_missing": False,
    "warned_eval_boundary": False,
}
_DEFAULT_SPECTRAL_MAX_DIM = 128


@dataclass(frozen=True)
class DeltaWeightSnapshot:
    layer_idx: int
    matrix_type: str
    module_name: str
    weight: torch.Tensor
    shape: tuple[int, int]

    @property
    def key(self) -> tuple[str, int, str, tuple[int, int]]:
        return (self.module_name, self.layer_idx, self.matrix_type, self.shape)


def _spectral_probe_matrix(matrix: torch.Tensor, max_dim: int) -> torch.Tensor:
    """Return a bounded submatrix for spectral diagnostics.

    Full SVD on every selected training matrix is too expensive for in-run
    monitoring. Keep exact spectra for small matrices and use deterministic
    evenly spaced rows/cols for larger ones so the metric remains cheap and
    comparable over time.
    """
    if max_dim <= 0 or max(matrix.shape) <= max_dim:
        return matrix

    row_count, col_count = matrix.shape
    if row_count > max_dim:
        rows = torch.linspace(0, row_count - 1, steps=max_dim, device=matrix.device).round().long()
        matrix = matrix.index_select(0, rows)
    if col_count > max_dim:
        cols = torch.linspace(0, col_count - 1, steps=max_dim, device=matrix.device).round().long()
        matrix = matrix.index_select(1, cols)
    return matrix


def compute_delta_w_stats(
    before,
    after,
    eps: float = 1e-12,
    spectral_max_dim: int = _DEFAULT_SPECTRAL_MAX_DIM,
) -> dict[str, float]:
    """Compute scalar diagnostics for a realized 2-D weight displacement."""
    w_before = before.detach().to(torch.float32)
    w_after = after.detach().to(torch.float32)
    delta = w_after - w_before
    out_dim, in_dim = delta.shape

    w_fro_before = torch.linalg.vector_norm(w_before).item()
    w_fro_after = torch.linalg.vector_norm(w_after).item()
    fro_abs = torch.linalg.vector_norm(delta).item()
    fro_rel = fro_abs / max(w_fro_before, eps)
    if fro_abs <= eps:
        w_fro_ratio = (
            1.0 if torch.equal(w_before, w_after) else w_fro_after / max(w_fro_before, eps)
        )
        return {
            "fro_abs": 0.0,
            "fro_rel": 0.0,
            "w_fro_before": float(w_fro_before),
            "w_fro_after": float(w_fro_after),
            "w_fro_ratio": float(w_fro_ratio),
            "cos_to_w": 0.0,
            "row_rms_delta_mean": 0.0,
            "col_rms_delta_mean": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "condition_number": 0.0,
            "sigma_max_over_median": 0.0,
            "stable_rank_frac": 0.0,
            "effective_rank_frac": 0.0,
        }

    w_fro_ratio = w_fro_after / max(w_fro_before, eps)
    denom = fro_abs * max(w_fro_before, eps)
    cos_to_w = torch.sum(delta * w_before).item() / denom if w_fro_before > eps else 0.0

    row_rms_delta_mean = torch.linalg.vector_norm(delta, dim=1).div(math.sqrt(in_dim)).mean().item()
    col_rms_delta_mean = (
        torch.linalg.vector_norm(delta, dim=0).div(math.sqrt(out_dim)).mean().item()
    )
    spectral_input = _spectral_probe_matrix(delta, int(spectral_max_dim or 0))
    spectral_min_dim = max(min(spectral_input.shape), 1)
    spectral = {k: v[0].item() for k, v in block_spectral_stats(spectral_input, eps=eps).items()}
    stable_rank = float(spectral["stable_rank"])
    effective_rank = float(spectral["effective_rank"])
    return {
        "fro_abs": float(fro_abs),
        "fro_rel": float(fro_rel),
        "w_fro_before": float(w_fro_before),
        "w_fro_after": float(w_fro_after),
        "w_fro_ratio": float(w_fro_ratio),
        "cos_to_w": float(cos_to_w),
        "row_rms_delta_mean": float(row_rms_delta_mean),
        "col_rms_delta_mean": float(col_rms_delta_mean),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "condition_number": float(spectral["condition_number"]),
        "sigma_max_over_median": float(spectral["sigma_max_over_median"]),
        "stable_rank_frac": stable_rank / float(spectral_min_dim),
        "effective_rank_frac": effective_rank / float(spectral_min_dim),
    }


def snapshot_target_weights(
    model,
    selected_layers: set[int],
    max_targets: int = 0,
) -> list[DeltaWeightSnapshot]:
    """Snapshot selected transformer weights as CPU float32 clones."""
    chunks = model if isinstance(model, list) else [model]
    out: list[DeltaWeightSnapshot] = []
    cap = int(max_targets or 0)
    with torch.no_grad():
        for chunk in chunks:
            for name, mod in chunk.named_modules():
                cls = classify_linear(name)
                if cls is None:
                    continue
                layer_idx, matrix_type = cls
                if layer_idx not in selected_layers:
                    continue
                weight = getattr(mod, "weight", None)
                if weight is None or getattr(weight, "dim", lambda: 0)() != 2:
                    continue
                clone = weight.detach().to(device="cpu", dtype=torch.float32).clone()
                out.append(
                    DeltaWeightSnapshot(
                        layer_idx=layer_idx,
                        matrix_type=matrix_type,
                        module_name=name,
                        weight=clone,
                        shape=tuple(clone.shape),
                    )
                )
                if cap > 0 and len(out) >= cap:
                    return out
    return out


def _wandb_run_active() -> bool:
    try:
        import wandb
    except Exception:
        return False
    return getattr(wandb, "run", None) is not None


def _log_delta_w_snapshots(
    before: list[DeltaWeightSnapshot],
    after: list[DeltaWeightSnapshot],
    iteration: int,
    spectral_max_dim: int = _DEFAULT_SPECTRAL_MAX_DIM,
) -> None:
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return

    after_by_key = {snap.key: snap for snap in after}
    payload: dict[str, float] = {}
    means: dict[str, list[float]] = {}
    for prev in before:
        curr = after_by_key.get(prev.key)
        if curr is None:
            if not _state["warned_missing"]:
                logger.warning(
                    "[DW] target disappeared or changed shape before after-snapshot; "
                    "skipping missing delta-W targets."
                )
                _state["warned_missing"] = True
            continue
        stats = compute_delta_w_stats(prev.weight, curr.weight, spectral_max_dim=spectral_max_dim)
        prefix = f"deltaw/L{prev.layer_idx}/{prev.matrix_type}"
        for metric, value in stats.items():
            payload[f"{prefix}/{metric}"] = value
            means.setdefault(metric, []).append(value)

    if not payload:
        return
    for metric, values in means.items():
        payload[f"deltaw/_mean/{metric}"] = float(sum(values) / len(values))
    wandb.log(payload, step=iteration)


def _resolve_iteration(args, kwargs, opts) -> int:
    iteration = kwargs.get("iteration")
    if iteration is None and len(args) >= 8:
        iteration = args[7]
    if iteration is None:
        iteration = getattr(opts, "iteration", 0)
    return int(iteration or 0)


def _model_from_train_step_args(args, kwargs):
    return args[2] if len(args) >= 3 else kwargs.get("model")


def _is_distributed_eval_boundary(opts, iteration: int) -> bool:
    if bool(getattr(opts, "poet", False)):
        return False
    if not bool(getattr(opts, "use_distributed_optimizer", False)):
        return False
    if not bool(getattr(opts, "overlap_param_gather", False)):
        return False
    eval_interval = int(getattr(opts, "eval_interval", 0) or 0)
    return iteration > 0 and eval_interval > 0 and iteration % eval_interval == 0


def _prepare_snapshot(args, kwargs, opts) -> tuple[int, list[DeltaWeightSnapshot] | None]:
    if not getattr(opts, "log_delta_w", False):
        return (0, None)

    iteration = _resolve_iteration(args, kwargs, opts)
    interval = int(getattr(opts, "log_delta_w_interval", 250))
    poet = bool(getattr(opts, "poet", False))
    merge_period = int(getattr(opts, "poet_merge_period", 0))

    if poet and merge_period <= 0:
        if not _state["warned_merge0"]:
            logger.warning(
                "[DW] POET with merge_period=0: base weight is frozen; delta-W "
                "logging is a no-op (W_eff is not materialized)."
            )
            _state["warned_merge0"] = True
        return (iteration, None)

    if poet and merge_period > 0 and interval % merge_period != 0 and not _state["warned_cadence"]:
        eff = math.lcm(interval, merge_period)
        logger.warning(
            "[DW] POET log_delta_w_interval=%d is not a multiple of merge_period=%d; "
            "delta-W is logged only on merge boundaries, so the effective logging "
            "cadence is %d steps. Set the interval to a multiple of merge_period "
            "for the intended cadence.",
            interval,
            merge_period,
            eff,
        )
        _state["warned_cadence"] = True

    if not should_log(iteration, interval, poet=poet, merge_period=merge_period):
        return (iteration, None)
    if _is_distributed_eval_boundary(opts, iteration):
        if not _state["warned_eval_boundary"]:
            logger.warning(
                "[DW] skipping distributed-optimizer eval-boundary delta-W step; "
                "overlap-param-gather can leave materialized weights stale and log "
                "false zero deltas."
            )
            _state["warned_eval_boundary"] = True
        return (iteration, None)
    if not _wandb_run_active():
        return (iteration, None)

    tp_size = int(getattr(opts, "tensor_model_parallel_size", 1) or 1)
    if tp_size > 1 and not _state["warned_tp"]:
        logger.warning(
            "[DW] tensor_model_parallel_size=%d; logging local weight shards only.",
            tp_size,
        )
        _state["warned_tp"] = True

    model = _model_from_train_step_args(args, kwargs)
    if model is None:
        logger.warning("[DW] model not found in train_step args; skipping")
        return (iteration, None)

    num_layers = int(getattr(opts, "num_layers", 0) or 0)
    if num_layers <= 0:
        return (iteration, None)
    selected_layers = parse_layer_selection(
        getattr(opts, "delta_w_layers", "first,mid,last"), num_layers
    )
    if not selected_layers:
        return (iteration, None)
    before = snapshot_target_weights(
        model,
        selected_layers,
        max_targets=int(getattr(opts, "delta_w_max_targets", 0) or 0),
    )
    return (iteration, before or None)


def _wrapped_train_step_factory(orig_train_step, get_args=None):
    """Build the outer train_step wrapper. ``get_args`` is injectable for tests."""

    def _wrapped(*args, **kwargs):
        iteration = 0
        before = None
        try:
            _get_args = get_args
            if _get_args is None:
                from megatron.training import get_args as _get_args  # type: ignore
            opts = _get_args()
            iteration, before = _prepare_snapshot(args, kwargs, opts)
        except Exception:
            logger.exception("[DW] delta-W before-snapshot failed; continuing")

        ret = orig_train_step(*args, **kwargs)

        if before is None:
            return ret
        try:
            _get_args = get_args
            if _get_args is None:
                from megatron.training import get_args as _get_args  # type: ignore
            opts = _get_args()
            model = _model_from_train_step_args(args, kwargs)
            if model is None:
                logger.warning("[DW] model not found in train_step args after step; skipping")
                return ret
            num_layers = int(getattr(opts, "num_layers", 0) or 0)
            selected_layers = parse_layer_selection(
                getattr(opts, "delta_w_layers", "first,mid,last"), num_layers
            )
            after = snapshot_target_weights(
                model,
                selected_layers,
                max_targets=int(getattr(opts, "delta_w_max_targets", 0) or 0),
            )
            _log_delta_w_snapshots(
                before,
                after,
                iteration,
                spectral_max_dim=max(
                    0,
                    int(getattr(opts, "delta_w_spectral_max_dim", _DEFAULT_SPECTRAL_MAX_DIM) or 0),
                ),
            )
        except Exception:
            logger.exception("[DW] delta-W logging failed; continuing")
        return ret

    return _wrapped


@register_patch(name="weight_delta_monitor", targets=())
def apply() -> None:
    from megatron.training import training as _mt

    _mt.train_step = _wrapped_train_step_factory(_mt.train_step)
