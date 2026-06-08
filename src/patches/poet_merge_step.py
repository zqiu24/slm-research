"""Patch: periodic POET merge-and-reinitialize in the training loop.

Targets ``megatron.training.training.train_step``. After each step,
if ``args.poet`` is set and ``iteration % args.poet_merge_period == 0``,
calls ``POETLinear.merge_then_reinitialize()`` on every POET layer and
broadcasts the updated state across ranks.

The fork-2 equivalent called ``poet_check_and_merge(model, iter, gap)``
from inside the training loop body. We instead wrap ``train_step``, which
receives ``(forward_step_func, data_iterator, model, optimizer,
opt_param_scheduler, config, forward_backward_func, iteration=None)`` —
``model`` is the 3rd positional arg, ``iteration`` the 8th kwarg/positional.

Merge correctness across the Megatron optimizer (the GaLore reference is plain
PyTorch AdamW, so it has none of this):

* The merge folds the current rotation ``R(oft_R)`` into the *frozen* base
  weight and zeros the **bf16 model** ``oft_R``. In a mixed-precision / sharded
  optimizer the parameter the optimizer actually steps is the **fp32 master**
  copy, not the bf16 model tensor. If only the model tensor is zeroed, the next
  ``optimizer.step()`` copies the still-nonzero master back into the model and
  ``oft_R`` *springs back to its pre-merge value* — re-applying the rotation a
  second time on top of the already-merged weight → huge recurring loss spike
  every ``poet_merge_period`` steps. So we must zero the master VALUE too.
* The post-merge Adam-momentum reset must also reach those masters. The plain
  ``Float16OptimizerWithFloat16Params`` exposes ``float16_groups`` /
  ``fp32_from_float16_groups``; the ``DistributedOptimizer`` (used whenever
  ``distributed_optimizer: true``, e.g. cluster=h100_de) instead exposes
  ``model_float16_groups`` / ``shard_fp32_from_float16_groups`` (sharded
  masters). The reset must handle BOTH layouts, or it silently resets zero
  params on the distributed path.

``_reset_vanilla_oft_state`` below handles both for the default Megatron-Adam
path (``optim.poet.use_poet_adam=false``). The custom POETAdam path resets its
own momentum; it does not yet zero the master value (tracked separately).
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.train_step",)
logger = logging.getLogger(__name__)


def _merge_decision(iteration: int, merge_period: int, reinit_period: int) -> tuple[bool, bool]:
    """Decide, for ``iteration``, whether to fold and whether to also reinit.

    Returns ``(folding, reinit)``:

    * ``folding`` — fold ``R(Q)`` into ``W`` and reset ``Q`` this step (cadence
      ``merge_period``). poet0 sets ``merge_period=1`` → fold every step.
    * ``reinit`` — *additionally* resample the block permutation Ψ and reset Adam
      momentum. Cadence by ``reinit_period``:
        - ``> 0``: reinit every ``reinit_period`` steps (must be a multiple of
          ``merge_period``; validated at arg-build time in megatron_args).
        - ``== 0``: fall back to ``merge_period`` (legacy fused behavior — reinit
          on every fold).
        - ``< 0``: NEVER reinit — constant fold with persistent momentum and a
          fixed Ψ. Intended for ``block_count=1`` (one block = the full matrix, so
          permutation resampling adds no coverage and only churns momentum).
      A reinit can only happen on a step that also folds.
    """
    if merge_period <= 0 or iteration <= 0 or iteration % merge_period != 0:
        return (False, False)
    if reinit_period < 0:
        return (True, False)
    gap = reinit_period if reinit_period > 0 else merge_period
    return (True, iteration % gap == 0)


def _seed_active_side(iteration: int) -> None:
    """Seed the shared active-side signal so the layer forward, optimizer step, and
    merge all read the same side within this training step."""
    from poet_torch.alt_state import set_iteration

    set_iteration(int(iteration) if iteration is not None else 0)


@register_patch(name="poet_merge_step", targets=_TARGET)
def apply() -> None:
    import torch.distributed as dist
    from megatron.training import get_args
    from megatron.training import training as _mt

    _orig_train_step = _mt.train_step

    def _wrapped(*args, **kwargs):
        opts = get_args()
        if not getattr(opts, "poet", False):
            return _orig_train_step(*args, **kwargs)
        iteration = kwargs.get("iteration")
        if iteration is None and len(args) >= 8:
            iteration = args[7]
        if iteration is None:
            iteration = getattr(opts, "iteration", 0)
        # Seed the active-side signal BEFORE forward so the layer reads this step's side.
        _seed_active_side(iteration)
        ret = _orig_train_step(*args, **kwargs)
        merge_period = getattr(opts, "poet_merge_period", 0)
        reinit_period = getattr(opts, "poet_reinit_period", 0)
        folding, do_reinit = _merge_decision(iteration, merge_period, reinit_period)
        if not folding:
            return ret
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if model is None:
            logger.warning("[POET] merge step skipped: model not found in train_step args")
            return ret
        _run_merge(model, dist, iteration, reinit_perm=do_reinit)
        # Megatron-Adam path (default): reset momentum ONLY when Ψ is resampled
        # (do_reinit) — otherwise momentum persists across the per-step fold. The
        # master VALUE is zeroed every fold regardless (inside _reset_vanilla_oft_state).
        if not getattr(opts, "poet_use_poet_adam", False):
            optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
            if optimizer is not None:
                _reset_vanilla_oft_state(optimizer, model, iteration, reset_moments=do_reinit)
        return ret

    _mt.train_step = _wrapped


def _iter_model_master_pairs(opt):
    """Yield ``(model_param, master_param)`` for one inner Megatron optimizer.

    Covers every mixed-precision layout so the same caller works on single-GPU
    and multi-GPU runs:

    * ``Float16OptimizerWithFloat16Params`` (distributed_optimizer=false, e.g.
      cluster=dev / single GPU): ``float16_groups`` <-> ``fp32_from_float16_groups``;
      master is a full fp32 copy.
    * ``DistributedOptimizer`` (distributed_optimizer=true, e.g. cluster=h100_de;
      also used at DP=1): ``model_float16_groups`` <-> ``shard_fp32_from_float16_groups``;
      master is *this rank's* fp32 shard of the param.
    * ``FP32Optimizer``: no float16 groups — the optimizer steps the model param
      directly, so model IS master.

    ``master_param`` may equal ``model_param`` (fp32 path) and is never None.
    """
    # (1) plain mixed-precision, then (2) distributed (sharded) layout.
    f16 = getattr(opt, "float16_groups", None)
    m32 = getattr(opt, "fp32_from_float16_groups", None)
    if f16 is None or m32 is None:
        f16 = getattr(opt, "model_float16_groups", None)
        m32 = getattr(opt, "shard_fp32_from_float16_groups", None)

    if f16 is not None and m32 is not None:
        for f16_grp, master_grp in zip(f16, m32, strict=False):
            for model_p, master_p in zip(f16_grp, master_grp, strict=False):
                if master_p is not None:
                    yield model_p, master_p
        return

    # FP32Optimizer: master == model.
    torch_opt = getattr(opt, "optimizer", None)
    if torch_opt is not None:
        for group in torch_opt.param_groups:
            for p in group["params"]:
                yield p, p


def _reset_vanilla_oft_state(optimizer, model, iteration: int, reset_moments: bool = True) -> None:
    """POETAdam-faithful per-merge reset for the Megatron-Adam POET path (default).

    For the oft_R params ONLY (leaving embedding/norm state untouched), this:

    * zeros the fp32 *master* value so it matches the merge's zeroed bf16 model
      tensor and cannot spring back on the next optimizer step (which would
      re-apply the just-merged rotation a second time -> loss spike);
    * zeros the master's Adam moments (``exp_avg`` / ``exp_avg_sq`` / per-param
      and per-group ``step``) so the post-merge restart gets fresh momentum and
      bias correction.

    Both single-GPU (plain Float16 optimizer, full master) and multi-GPU
    (DistributedOptimizer, sharded master) are covered by
    ``_iter_model_master_pairs``.
    """
    import torch

    chunks = model if isinstance(model, list) else [model]
    oft_ids = {
        id(p)
        for m in chunks
        for name, p in m.named_parameters()
        if "oft_R" in name and p.requires_grad
    }

    def _zero_moments(master_param, torch_opt) -> int:
        st = torch_opt.state.get(master_param)
        if not st:
            return 0
        if "exp_avg" in st:
            st["exp_avg"].zero_()
        if "exp_avg_sq" in st:
            st["exp_avg_sq"].zero_()
        if "step" in st:
            if torch.is_tensor(st["step"]):
                st["step"].zero_()
            else:
                st["step"] = 0
        return 1

    inner = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    oft_master_ids = set()
    seen_opts = []

    for opt in inner:
        torch_opt = getattr(opt, "optimizer", None)
        if torch_opt is None:
            continue
        if torch_opt not in seen_opts:
            seen_opts.append(torch_opt)
        for model_p, master_p in _iter_model_master_pairs(opt):
            if id(model_p) not in oft_ids:
                continue
            oft_master_ids.add(id(master_p))
            # Zero the fp32 master VALUE (no-op if master IS the model tensor,
            # which the merge already zeroed). ALWAYS done — load-bearing against
            # spring-back of the just-merged rotation.
            if master_p is not model_p:
                master_p.detach().zero_()
            # Moments reset only when reinit fires (Ψ changed -> new coordinate
            # frame). poet0 non-boundary steps keep momentum (reset_moments=False).
            if reset_moments:
                _zero_moments(master_p, torch_opt)

    # This Megatron Adam stores ``step`` PER param-group (not per-param), so the
    # per-param reset above doesn't refresh bias correction. Reset the group-level
    # step for any group holding oft_R masters so t -> 0 and the post-merge
    # restart gets fresh bias correction.
    if reset_moments:
        for torch_opt in seen_opts:
            for group in torch_opt.param_groups:
                if "step" not in group:
                    continue
                if not any(id(p) in oft_master_ids for p in group["params"]):
                    continue
                if torch.is_tensor(group["step"]):
                    group["step"].zero_()
                else:
                    group["step"] = 0


def _build_R_batched(layers, cayley_fn=None, max_batch_block: int = 256):
    """Build (R_out, R_in) for every layer, batching the Cayley across layers that
    share a block size on a side (small blocks only). Returns {id(layer): (R_out, R_in)}.

    Cayley acts independently per [b,b] block, so concatenating blocks across layers
    and running one kernel is bit-identical to per-layer calls. Sides with block_size
    > max_batch_block (e.g. block_count=1 dense) are built per-layer to bound the
    transient memory of stacking big blocks.

    cayley_fn(Q[*, b, b]) -> R[*, b, b]; defaults to the Triton op. Tests inject the
    pure-torch cayley_batch.
    """
    import torch
    from poet_torch.poet_layer import pytorch_skew_symmetric

    if cayley_fn is None:

        def cayley_fn(Q):
            return torch.ops.poet.cayley(Q)[0]

    result = {id(pl): [None, None] for pl in layers}  # [R_out, R_in]
    # side_idx 0 -> out, 1 -> in
    for side_idx, side in enumerate(("out", "in")):
        groups = {}  # block_size -> list of (pl, oft, rows, cols)
        for pl in layers:
            if side == "out":
                b, oft, rows, cols = pl.block_size_out, pl.oft_R_out, pl.rows_out, pl.cols_out
            else:
                b, oft, rows, cols = pl.block_size_in, pl.oft_R_in, pl.rows_in, pl.cols_in
            groups.setdefault(int(b), []).append((pl, oft, rows, cols))
        for b, items in groups.items():
            if b <= max_batch_block and len(items) > 1:
                rows, cols = items[0][2], items[0][3]
                skews = [pytorch_skew_symmetric(oft, b, rows, cols) for (_, oft, _, _) in items]
                sizes = [s.shape[0] for s in skews]
                R = cayley_fn(torch.cat(skews, dim=0))  # ONE Cayley for the whole group
                off = 0
                for (pl, _, _, _), n in zip(items, sizes, strict=True):
                    result[id(pl)][side_idx] = R[off : off + n]
                    off += n
            else:
                for pl, oft, rows, cols in items:
                    R = cayley_fn(pytorch_skew_symmetric(oft, b, rows, cols))
                    result[id(pl)][side_idx] = R
    return {k: (v[0], v[1]) for k, v in result.items()}


# Module-level: the one-time perm sync flag for the replicate path (see below).
_perms_synced = False


def _run_merge(model, dist, iteration: int, reinit_perm: bool = True) -> None:
    import os

    import torch
    from poet_torch import POETLinear, POETXLinear

    from src.optim.poet_layers import POETMegatronLinear

    global _perms_synced

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # Collect the POET layers to merge (same filter as before).
    pls = []
    chunks = model if isinstance(model, list) else [model]
    for m in chunks:
        for _, mod in m.named_modules():
            if not isinstance(mod, POETMegatronLinear):
                continue
            pl = mod.poet_linear
            if not isinstance(pl, POETLinear | POETXLinear) or pl.block_size <= 0:
                continue
            pls.append(pl)

    # Escape hatches (debugging only): force the legacy rank-0 + broadcast path,
    # and/or disable Cayley batching.
    force_broadcast = os.environ.get("POET_FORCE_MERGE_BROADCAST") == "1"
    disable_batch = os.environ.get("POET_DISABLE_BATCHED_MERGE") == "1"

    # REPLICATE: when permutations are NOT being resampled, the fold is a
    # deterministic function of DP-identical (oft_R, W), so every rank folds its
    # own replica to a bit-identical result with NO communication (same reason DDP
    # never broadcasts weights). reinit_perm=True (randperm) is rank-divergent, so
    # fall back to rank-0 + broadcast for that (rare/disabled) case.
    replicate = (not reinit_perm) and (not force_broadcast)

    if replicate:
        # One-time perm sync: guarantee DP-identical permutations before trusting
        # determinism. randperm-at-init *should* match across DP (identical model
        # seed), but sync once to be certain — without it, divergent perms would
        # silently diverge W. No-op if already identical; perms never change after
        # (reinit_period<0), so this runs at most once per process.
        if is_dist and not _perms_synced:
            for pl in pls:
                for buf in (pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv):
                    dist.broadcast(buf, src=0)
            _perms_synced = True
        with torch.no_grad():
            _merge_layers(pls, reinit_perm=False, disable_batch=disable_batch)
        # Debug gate (off by default): verify the no-broadcast replicate fold keeps
        # every DP rank's frozen W bit-identical. Acceptance is drift == 0.0 every
        # step; non-zero means a non-deterministic kernel or desynced perms.
        if os.environ.get("POET_CHECK_MERGE_SYNC") == "1" and is_dist and pls:
            w = pls[0].weight.data.clone()
            ref = w.clone()
            dist.broadcast(ref, src=0)
            drift = (w - ref).abs().max()
            if rank == 0:
                print(
                    f"[POET] merge cross-rank drift (rank-vs-0): {drift.item():.2e}",
                    flush=True,
                )
        for pl in pls:
            if hasattr(pl, "_invalidate_R_cache"):
                pl._invalidate_R_cache()
        return

    # Legacy path: rank-0 folds, then broadcast (covers reinit_perm=True and the
    # forced escape hatch).
    with torch.no_grad():
        if rank == 0:
            _merge_layers(pls, reinit_perm=reinit_perm, disable_batch=disable_batch)
        if is_dist:
            for pl in pls:
                for buf in (
                    pl.oft_R_in.data,
                    pl.oft_R_out.data,
                    pl.weight.data,
                    pl.perm_in,
                    pl.perm_in_inv,
                    pl.perm_out,
                    pl.perm_out_inv,
                ):
                    dist.broadcast(buf, src=0)
    for pl in pls:
        if hasattr(pl, "_invalidate_R_cache"):
            pl._invalidate_R_cache()


def _merge_layers(pls, reinit_perm: bool, disable_batch: bool) -> None:
    """Fold every layer. AlternatingPOETXLinear layers fold ONLY the active side
    (frozen side is identity); the rest use the batched both-sides fold. The active
    side comes from each layer's OWN alternate_every via alt_state — no megatron
    get_args, so _merge_layers stays importable/callable on CPU (megatron is not
    importable in the unit-test venv)."""
    from poet_torch import AlternatingPOETXLinear
    from poet_torch.alt_state import active_side

    alt_pls = [pl for pl in pls if isinstance(pl, AlternatingPOETXLinear)]
    rest = [pl for pl in pls if not isinstance(pl, AlternatingPOETXLinear)]

    for pl in alt_pls:
        pl._fold_active_side(active_side(pl.alternate_every), reinit_perm=reinit_perm)

    if disable_batch:
        for pl in rest:
            pl.merge_then_reinitialize(reinit_perm=reinit_perm)
        return
    cayley_pls = [pl for pl in rest if getattr(pl, "parameterization", "cayley") == "cayley"]
    other_pls = [pl for pl in rest if getattr(pl, "parameterization", "cayley") != "cayley"]
    for pl in other_pls:
        pl.merge_then_reinitialize(reinit_perm=reinit_perm)
    if cayley_pls:
        built = _build_R_batched(cayley_pls)  # default cayley_fn = Triton op
        for pl in cayley_pls:
            R_out, R_in = built[id(pl)]
            pl._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
