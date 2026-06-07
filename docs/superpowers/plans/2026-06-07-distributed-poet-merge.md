# Distributed POET Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-step POET merge cheaper at high data-parallel (DP) world size by adding two opt-in merge modes — `comm_free` (every rank merges itself, drop the broadcast) and `sharded` (round-robin owners, broadcast results over the DP group) — selected by a config flag, with the default preserving today's exact behavior.

**Architecture:** The merge runs after `train_step`, when every DP rank already holds bit-identical `weight`/`oft_R` (the `DistributedOptimizer` all-gathers params at the end of the step). The merge is a deterministic pure function of those tensors, so the current rank-0-compute-then-broadcast is one of three strategies: **A** current (`T_merge + C`), **B** `comm_free` (`T_merge`, no comm), **C** `sharded` (`T_merge/dp + C`). Measurement picks B (comm-bound, expected at Kimi-1T scale) or C (compute-bound). Task 1 adds env-gated measurement scaffolding; **the user runs the measurement and the decision gate before B/C are wired on.**

**Tech Stack:** Python, PyTorch (`torch.distributed`), Megatron-Core (`megatron.core.parallel_state` as `mpu`), Hydra/OmegaConf config, the repo's `register_patch` system, pytest (CPU unit tests).

**Spec:** [`docs/superpowers/specs/2026-06-07-distributed-poet-merge-design.md`](../specs/2026-06-07-distributed-poet-merge-design.md)

---

## Conventions for this plan

- **CPU tests** (Tasks 2, 3): run with the repo's test venv —
  `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest …`. The base `python` lacks `omegaconf`/`torch`.
- **GPU / distributed validation** (Tasks 1, 5, 6, 7): multi-GPU cluster runs are **yours to launch**. Each such task ends with the exact command and what to look for; the agent does **not** run them.
- **Commits:** conventional-commit prefixes matching the repo (`feat(poet):`, `test(poet):`, `refactor(poet):`). One short line, no AI attribution.
- **File under edit:** the merge driver is [`src/patches/poet_merge_step.py`](../../../src/patches/poet_merge_step.py). Read it once before starting — Tasks 1, 4, 5, 6 all edit it.

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/patches/poet_merge_step.py` | merge driver: layer iteration, env-gated profiling/verify, mode dispatch, the three merge strategies | 1, 4, 5, 6 |
| `src/optim/poet_merge_dist.py` (new) | **pure** DP-ownership helpers (round-robin, cost-aware), zero torch/megatron imports → CPU-unit-testable | 2 |
| `launchers/pretrain_gpt_slm.py` | argparse registration of `--poet-merge-distributed` / `--poet-merge-reanchor-period` | 3 |
| `src/utils/megatron_args.py` | emit those flags from `optim.poet.*` YAML | 3 |
| `tests/unit/test_poet_merge_dist.py` (new) | ownership-helper unit tests | 2 |
| `tests/unit/test_pretrain_gpt_slm.py` | argparse-acceptance test | 3 |
| `tests/unit/test_megatron_args.py` | YAML→argv emission test | 3 |

---

## Task 1: Measurement scaffolding + DP-identity verify (env-gated)

Adds the `§4` instrumentation to the **existing** merge (mode A behavior unchanged in result): per-phase timers (`compute` vs `comm` vs `step`) and a cross-DP checksum that proves B's invariant. Everything is gated behind env vars and off by default. Restructures `_run_merge` into a `compute`-phase then `comm`-phase (result-identical, since each layer's merge is independent) so the two phases can be timed with one sync each. Also extracts a `_poet_layers` generator reused by later tasks.

**Files:**
- Modify: `src/patches/poet_merge_step.py`

This task is GPU/distributed-validated (no CPU unit test). The validation is the bit-exactness check in Step 4 + the profile run.

- [ ] **Step 1: Add module-level profiling/verify state and helpers**

Insert after the imports block (after `logger = logging.getLogger(__name__)`, ~line 45):

```python
import contextlib
import os
import time

# --- env-gated measurement scaffolding (off unless explicitly enabled) ---------
_MERGE_PROFILE = os.environ.get("POET_MERGE_PROFILE") == "1"
_MERGE_VERIFY = os.environ.get("POET_MERGE_VERIFY") == "1"
_MERGE_REPORT_EVERY = int(os.environ.get("POET_MERGE_REPORT_EVERY", "20"))
_PROF = {"compute_ms": 0.0, "comm_ms": 0.0, "step_ms": 0.0, "steps": 0}


@contextlib.contextmanager
def _timer(which: str):
    """Accumulate wall-clock ms (with a CUDA sync on each boundary) into _PROF.

    No-op unless POET_MERGE_PROFILE=1, so it is free on production runs.
    """
    if not _MERGE_PROFILE:
        yield
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _PROF[f"{which}_ms"] += (time.perf_counter() - t0) * 1e3


def _dp_world_and_group(dist):
    """Return (world_size, group, rank) for the DP group, or (1, None, 0)."""
    try:
        from megatron.core import parallel_state as mpu

        return (
            mpu.get_data_parallel_world_size(),
            mpu.get_data_parallel_group(),
            mpu.get_data_parallel_rank(),
        )
    except Exception:  # pragma: no cover - non-megatron / uninitialized
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), None, dist.get_rank()
        return 1, None, 0


def _maybe_report(iteration: int, dist) -> None:
    """Every _MERGE_REPORT_EVERY steps, log mean compute/comm/step ms + the
    B-vs-C decision quantities (rank 0 only), then reset the accumulators."""
    if not _MERGE_PROFILE:
        return
    if iteration <= 0 or _MERGE_REPORT_EVERY <= 0 or iteration % _MERGE_REPORT_EVERY != 0:
        return
    dp_world, _group, dp_rank = _dp_world_and_group(dist)
    n = max(_PROF["steps"], 1)
    compute = _PROF["compute_ms"] / n
    comm = _PROF["comm_ms"] / n
    step = _PROF["step_ms"] / n
    if dp_rank == 0:
        merge_frac = (compute + comm) / step if step else 0.0
        b_save = comm / step if step else 0.0
        c_save = (compute * (1.0 - 1.0 / dp_world)) / step if (step and dp_world > 1) else 0.0
        logger.info(
            "[POET-merge-profile] iter=%d dp=%d compute=%.3fms comm=%.3fms step=%.3fms "
            "merge_frac=%.4f B_save=%.4f C_save=%.4f",
            iteration, dp_world, compute, comm, step, merge_frac, b_save, c_save,
        )
    _PROF.update(compute_ms=0.0, comm_ms=0.0, step_ms=0.0, steps=0)


def _poet_layers(model):
    """Yield every active POETLinear (block_size>0) across model chunks, in a
    deterministic, rank-identical order (named_modules walk)."""
    from poet_torch import POETLinear

    from src.optim.poet_layers import POETMegatronLinear

    chunks = model if isinstance(model, list) else [model]
    for m in chunks:
        for _, mod in m.named_modules():
            if not isinstance(mod, POETMegatronLinear):
                continue
            pl = mod.poet_linear
            if not isinstance(pl, POETLinear) or pl.block_size <= 0:
                continue
            yield pl


def _verify_dp_identical(layers, dist) -> None:
    """Assert every rank's pre-merge POET weight/oft_R are bit-identical across
    the DP group (the invariant Approach B relies on). POET_MERGE_VERIFY=1."""
    import torch

    dp_world, group, _rank = _dp_world_and_group(dist)
    if dp_world <= 1 or not (dist.is_available() and dist.is_initialized()):
        return
    vals = []
    for pl in layers:
        for t in (pl.weight, pl.oft_R_in, pl.oft_R_out):
            d = t.detach().double()
            vals.append(d.sum())
            vals.append(d.abs().sum())
    if not vals:
        return
    v = torch.stack(vals)
    vmax = v.clone()
    vmin = v.clone()
    dist.all_reduce(vmax, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(vmin, op=dist.ReduceOp.MIN, group=group)
    if not torch.equal(vmax, vmin):
        n_diff = int((vmax != vmin).sum().item())
        raise AssertionError(
            f"[POET-merge-verify] pre-merge POET state diverges across DP "
            f"({n_diff} checksum entries differ) — comm_free (Approach B) is UNSAFE here."
        )
```

- [ ] **Step 2: Rewrite `_run_merge` into timed compute/comm phases (result-identical)**

Replace the body of `_run_merge` (currently ~lines 235-271) with:

```python
def _run_merge(model, dist, iteration: int, reinit_perm: bool = True) -> None:
    import torch

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    layers = list(_poet_layers(model))
    if _MERGE_VERIFY:
        _verify_dp_identical(layers, dist)

    # Phase 1: compute (rank 0 only, as today). Each layer's merge is independent,
    # so folding all layers before broadcasting is bit-identical to the old
    # interleaved fold/broadcast-per-layer loop.
    with _timer("compute"):
        with torch.no_grad():
            if rank == 0:
                for pl in layers:
                    pl.merge_then_reinitialize(reinit_perm=reinit_perm)

    # Phase 2: broadcast rank 0's result to all ranks (as today).
    with _timer("comm"):
        if is_dist:
            for pl in layers:
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

    for pl in layers:
        if hasattr(pl, "_invalidate_R_cache"):
            pl._invalidate_R_cache()
```

- [ ] **Step 3: Time the whole step + emit the periodic report in `_wrapped`**

Replace the entire `_wrapped` function inside `apply()` (~lines 82-109) with this complete version — it wraps the body in a `step` timer, counts folding steps, and reports periodically. (The merge-decision logic is unchanged; only the `with _timer("step"):` wrapper, the `_PROF["steps"]` increment, and the trailing `_maybe_report` are new.)

```python
    def _wrapped(*args, **kwargs):
        with _timer("step"):
            ret = _orig_train_step(*args, **kwargs)
            opts = get_args()
            if not getattr(opts, "poet", False):
                return ret
            merge_period = getattr(opts, "poet_merge_period", 0)
            reinit_period = getattr(opts, "poet_reinit_period", 0)
            iteration = kwargs.get("iteration")
            if iteration is None and len(args) >= 8:
                iteration = args[7]
            if iteration is None:
                iteration = getattr(opts, "iteration", 0)
            folding, do_reinit = _merge_decision(iteration, merge_period, reinit_period)
            if not folding:
                return ret
            model = args[2] if len(args) >= 3 else kwargs.get("model")
            if model is None:
                logger.warning("[POET] merge step skipped: model not found in train_step args")
                return ret
            if _MERGE_PROFILE:
                _PROF["steps"] += 1
            _run_merge(model, dist, iteration, reinit_perm=do_reinit)
            # Megatron-Adam path (default): reset momentum ONLY on a reinit boundary;
            # the master VALUE is zeroed every fold inside _reset_vanilla_oft_state.
            if not getattr(opts, "poet_use_poet_adam", False):
                optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
                if optimizer is not None:
                    _reset_vanilla_oft_state(optimizer, model, iteration, reset_moments=do_reinit)
        _maybe_report(iteration, dist)
        return ret
```

> Notes: returning from inside the `with` still records `step_ms` (the context manager's `__exit__` runs on return). `_maybe_report` is reached only on the folding path, which is every step at the target `merge_period=1` — the profiler assumes `merge_period=1` (mean would be diluted otherwise, since `step_ms` accrues on every call but `compute`/`comm`/`steps` only on folding steps).

- [ ] **Step 4: Validate mode-A result is unchanged (bit-exactness vs original)**

This restructure must not change A's output. Validate on a short single-GPU run by comparing a checksum of a POET `weight` after N steps against the pre-Task-1 commit.

Run (yours to launch):
```bash
codexlog merge_taskA_ref bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 training.train_iters=20
# then check the run's final loss / a weight checksum matches a run from HEAD~ (pre-Task-1).
```
Expected: identical loss trajectory to the pre-Task-1 baseline (merge result is deterministic; only the loop structure changed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py
git commit -m "feat(poet): env-gated merge profiling + DP-identity verify scaffolding"
```

---

> ## ⛔ GATE — run the §4 measurement before continuing
>
> With Task 1 committed, run the profile on your real recipe (mode is still A; no new flag needed):
> ```bash
> POET_MERGE_VERIFY=1 POET_MERGE_PROFILE=1 \
> codexlog lieorth_merge_profile bash scripts/train_poet_lie_orth.sh llama3 \
>   optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true
> ```
> Read `[POET-merge-profile]` lines from `/lustre/home/zqiu/log/lieorth_merge_profile.log`.
> **Decision (per spec §4):** `merge_frac < ~0.01-0.02` → stop (not worth it). `comm ≳ compute·(1−1/dp)` → ship **B** (Tasks 2-5, skip 6). Else → ship **C** (Tasks 2-6). A `[POET-merge-verify]` AssertionError means the B-invariant fails — resolve before B; C is still safe.
>
> Tasks 2-3 (ownership helpers + flag plumbing) are prerequisites for **both** B and C; build them once the gate says "go."

---

## Task 2: Pure DP-ownership helpers (CPU, TDD)

Round-robin and cost-aware layer→rank assignment as **pure functions** (lists of ints in, list of owner-ranks out). No torch/megatron imports, so they unit-test on CPU. Used by Approach C (Task 6); round-robin keeps it simple, cost-aware balances non-uniform POET layer shapes at high `dp`.

**Files:**
- Create: `src/optim/poet_merge_dist.py`
- Test: `tests/unit/test_poet_merge_dist.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_merge_dist.py`:

```python
from __future__ import annotations

from src.optim.poet_merge_dist import owner_round_robin, owners_cost_aware


def test_round_robin_cycles_by_index():
    assert owner_round_robin(5, 2) == [0, 1, 0, 1, 0]


def test_round_robin_single_rank_all_zero():
    assert owner_round_robin(4, 1) == [0, 0, 0, 0]


def test_cost_aware_balances_load():
    # Two big layers + two small: greedy puts one big on each rank.
    sizes = [100, 100, 1, 1]
    owners = owners_cost_aware(sizes, 2)
    load = [0, 0]
    for s, r in zip(sizes, owners):
        load[r] += s
    assert load[0] == load[1] == 101


def test_cost_aware_assigns_every_layer_a_valid_rank():
    owners = owners_cost_aware([7, 3, 5, 2, 9], 3)
    assert len(owners) == 5
    assert all(0 <= r < 3 for r in owners)


def test_cost_aware_single_rank_all_zero():
    assert owners_cost_aware([3, 1, 4], 1) == [0, 0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_dist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.optim.poet_merge_dist'`.

- [ ] **Step 3: Write the implementation**

Create `src/optim/poet_merge_dist.py`:

```python
"""Pure layer-to-rank ownership helpers for the distributed POET merge.

Kept import-light (no torch / megatron) so the assignment logic is unit-testable
on CPU. ``sizes`` are weight element counts; ``dp_world`` is the data-parallel
world size; the return is a list mapping layer index -> owner DP rank.
"""

from __future__ import annotations


def owner_round_robin(num_layers: int, dp_world: int) -> list[int]:
    """Assign layer ``i`` to rank ``i % dp_world`` (matches the lie_ortho q-update)."""
    if dp_world <= 1:
        return [0] * num_layers
    return [i % dp_world for i in range(num_layers)]


def owners_cost_aware(sizes: list[int], dp_world: int) -> list[int]:
    """Greedy longest-processing-time assignment: largest layers first, each to
    the currently least-loaded rank. Balances non-uniform POET layer shapes."""
    n = len(sizes)
    if dp_world <= 1:
        return [0] * n
    load = [0] * dp_world
    owners = [0] * n
    for i in sorted(range(n), key=lambda j: sizes[j], reverse=True):
        r = min(range(dp_world), key=lambda k: load[k])
        owners[i] = r
        load[r] += sizes[i]
    return owners
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_dist.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_merge_dist.py tests/unit/test_poet_merge_dist.py
git commit -m "feat(poet): pure DP-ownership helpers for distributed merge"
```

---

## Task 3: Flag plumbing — `merge_distributed` + `merge_reanchor_period` (CPU, TDD)

Register the new flags in argparse and emit them from YAML. `merge_distributed ∈ {off, comm_free, sharded}` (default `off` = today's behavior); `merge_reanchor_period` is the Approach-B drift-anchor cadence (0 = disabled). Mirrors the existing `lie_ortho_distributed` plumbing.

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (argparse, after the `--poet-lie-ortho-distributed` line ~102)
- Modify: `src/utils/megatron_args.py` (emit inside `poet_args`, after `--poet-reinit-period` ~line 297)
- Test: `tests/unit/test_pretrain_gpt_slm.py`, `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_merge_distributed():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    on = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--poet-merge-distributed", "comm_free"]
    )
    off = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert on.poet_merge_distributed == "comm_free"
    assert off.poet_merge_distributed == "off"


def test_add_slm_args_accepts_merge_reanchor_period():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    on = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--poet-merge-reanchor-period", "1000"]
    )
    off = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert on.poet_merge_reanchor_period == 1000
    assert off.poet_merge_reanchor_period == 0
```

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_emits_merge_distributed_default_off():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_count": 1, "merge_period": 1}))
    assert args[args.index("--poet-merge-distributed") + 1] == "off"


def test_poet_argv_emits_merge_distributed_comm_free():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "merge_distributed": "comm_free"})
    )
    assert args[args.index("--poet-merge-distributed") + 1] == "comm_free"


def test_poet_argv_emits_merge_reanchor_period():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "merge_reanchor_period": 1000})
    )
    assert args[args.index("--poet-merge-reanchor-period") + 1] == "1000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py -k merge \
  tests/unit/test_megatron_args.py -k merge_distributed -v
```
Expected: FAIL — `--poet-merge-distributed` unrecognized / not in emitted args.

- [ ] **Step 3: Register the argparse flags**

In `launchers/pretrain_gpt_slm.py`, immediately after the `--poet-lie-ortho-distributed` line (~102):

```python
    group.add_argument(
        "--poet-merge-distributed",
        choices=["off", "comm_free", "sharded"],
        default="off",
    )
    group.add_argument("--poet-merge-reanchor-period", type=int, default=0)
```

- [ ] **Step 4: Emit the flags from YAML**

In `src/utils/megatron_args.py`, inside the `poet_args` list, right after the
`"--poet-reinit-period", reinit_period,` pair (~line 297), add:

```python
            "--poet-merge-distributed",
            poet.get("merge_distributed", "off"),
            "--poet-merge-reanchor-period",
            poet.get("merge_reanchor_period", 0),
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py tests/unit/test_megatron_args.py -v
```
Expected: all pass (new tests green, existing poet tests still green).

- [ ] **Step 6: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py \
  tests/unit/test_pretrain_gpt_slm.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): plumb --poet-merge-distributed and --poet-merge-reanchor-period"
```

---

## Task 4: Mode dispatch in the merge driver

Read the flag in `_merge_body` and route `_run_merge` to the chosen strategy. `off` keeps the Task-1 (mode-A) path exactly. B/C land as stubs that raise `NotImplementedError` until Tasks 5/6 fill them — so this task is safe to commit on its own and the default path is untouched.

**Files:**
- Modify: `src/patches/poet_merge_step.py`

- [ ] **Step 1: Add the mode parameter and dispatch to `_run_merge`**

Change the `_run_merge` signature and split the strategy out. Rename the current Task-1 body into `_merge_rank0_broadcast(layers, dist, reinit_perm)` and make `_run_merge` dispatch:

```python
def _run_merge(model, dist, iteration: int, reinit_perm: bool = True,
               mode: str = "off", reanchor_period: int = 0) -> None:
    is_dist = dist.is_available() and dist.is_initialized()
    layers = list(_poet_layers(model))
    if _MERGE_VERIFY:
        _verify_dp_identical(layers, dist)

    if mode == "comm_free":
        _merge_comm_free(layers, dist, iteration, reinit_perm, reanchor_period)
    elif mode == "sharded":
        _merge_sharded(layers, dist, iteration, reinit_perm)
    else:  # "off" -> today's rank-0-compute + broadcast
        _merge_rank0_broadcast(layers, dist, reinit_perm)

    for pl in layers:
        if hasattr(pl, "_invalidate_R_cache"):
            pl._invalidate_R_cache()


def _merge_rank0_broadcast(layers, dist, reinit_perm: bool) -> None:
    import torch

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    with _timer("compute"):
        with torch.no_grad():
            if rank == 0:
                for pl in layers:
                    pl.merge_then_reinitialize(reinit_perm=reinit_perm)
    with _timer("comm"):
        if is_dist:
            for pl in layers:
                for buf in (
                    pl.oft_R_in.data, pl.oft_R_out.data, pl.weight.data,
                    pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv,
                ):
                    dist.broadcast(buf, src=0)


def _merge_comm_free(layers, dist, iteration, reinit_perm, reanchor_period):  # Task 5
    raise NotImplementedError("comm_free merge lands in Task 5")


def _merge_sharded(layers, dist, iteration, reinit_perm):  # Task 6
    raise NotImplementedError("sharded merge lands in Task 6")
```

(The `_verify_dp_identical` + cache-invalidation that lived in Task-1's `_run_merge` now live in the dispatcher above; remove them from the old inline body.)

- [ ] **Step 2: Read the flag in `_wrapped` and pass it through**

In `_wrapped` (Task 1 Step 3), where it reads merge config, add after `reinit_period = getattr(opts, "poet_reinit_period", 0)`:

```python
        merge_mode = getattr(opts, "poet_merge_distributed", "off")
        reanchor_period = getattr(opts, "poet_merge_reanchor_period", 0)
```

and change the merge call from:

```python
        _run_merge(model, dist, iteration, reinit_perm=do_reinit)
```

to:

```python
        _run_merge(model, dist, iteration, reinit_perm=do_reinit,
                   mode=merge_mode, reanchor_period=reanchor_period)
```

- [ ] **Step 3: Smoke-check default path still imports + behaves as A**

Run (CPU import smoke — patch module must import cleanly):
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -c "import src.patches.poet_merge_step as p; print('ok', hasattr(p, '_merge_comm_free'), hasattr(p, '_merge_sharded'))"
```
Expected: `ok True True`.

GPU default-path equivalence is covered by Task 7. With no `merge_distributed` set, behavior is identical to Task 1.

- [ ] **Step 4: Commit**

```bash
git add src/patches/poet_merge_step.py
git commit -m "refactor(poet): mode dispatch for merge (off|comm_free|sharded), B/C stubbed"
```

---

## Task 5: Approach B — comm-free redundant merge

Every rank merges every layer itself; no broadcast. Optional drift anchor re-broadcasts `weight` from rank 0 every `reanchor_period` steps. Guards against the one unsafe case: `reinit_perm=True` on >1 rank would draw divergent `randperm`s (shared-seed perms are deferred per spec §9), so raise a clear error there.

**Files:**
- Modify: `src/patches/poet_merge_step.py`

GPU/distributed-validated (Task 7).

- [ ] **Step 1: Implement `_merge_comm_free`**

Replace the Task-4 stub:

```python
def _merge_comm_free(layers, dist, iteration, reinit_perm, reanchor_period):
    """Approach B: every rank folds every layer itself (inputs are DP-identical,
    merge is deterministic) -> no per-step communication. Optional periodic
    re-broadcast from rank 0 bounds any FP drift."""
    import torch

    is_dist = dist.is_available() and dist.is_initialized()
    dp_world, group, _rank = _dp_world_and_group(dist)

    if reinit_perm and is_dist and dp_world > 1:
        raise NotImplementedError(
            "comm_free + reinit (Ψ resample) needs shared-seed permutations "
            "(spec §9, deferred). Use merge_distributed=sharded for reinit runs, "
            "or reinit_period=-1 (no resample) with comm_free."
        )

    with _timer("compute"):
        with torch.no_grad():
            for pl in layers:
                pl.merge_then_reinitialize(reinit_perm=reinit_perm)

    # Drift anchor: re-broadcast weights from rank 0 every reanchor_period steps.
    with _timer("comm"):
        if is_dist and dp_world > 1 and reanchor_period and iteration % reanchor_period == 0:
            for pl in layers:
                dist.broadcast(pl.weight.data, src=torch.distributed.get_global_rank(group, 0),
                               group=group)
```

> Why local zeroing is safe: `merge_then_reinitialize` zeros `oft_R` on every rank, and with `reinit_perm=False` the perms are untouched (already identical), so `weight` is the only changed tensor and every rank computes the identical new `weight` from identical inputs.

- [ ] **Step 2: GPU validation — loss parity vs mode A**

Run (yours): a short run with `comm_free` must track the mode-A loss curve and the verify check must pass.
```bash
POET_MERGE_VERIFY=1 \
codexlog merge_commfree bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.merge_distributed=comm_free
```
Expected: loss trajectory matches the mode-A baseline within run-to-run noise; no `[POET-merge-verify]` assertion; `[POET-merge-profile]` (if enabled) shows `comm≈0`.

- [ ] **Step 3: Commit**

```bash
git add src/patches/poet_merge_step.py
git commit -m "feat(poet): comm-free redundant merge (Approach B)"
```

---

## Task 6: Approach C — sharded compute + DP-group broadcast

Two phases: each rank folds only its owned layers (round-robin), zeroing `oft_R` locally on the rest; then each layer's `weight` (and perms on reinit) is broadcast from its owner over the **DP group** (not the world group — this also removes the latent TP>1 hazard).

**Files:**
- Modify: `src/patches/poet_merge_step.py`

GPU/distributed-validated (Task 7).

- [ ] **Step 1: Implement `_merge_sharded`**

Replace the Task-4 stub:

```python
def _merge_sharded(layers, dist, iteration, reinit_perm):
    """Approach C: round-robin owners fold their layers (parallel compute), then
    broadcast each layer's weight from its owner over the DP group."""
    import torch

    from src.optim.poet_merge_dist import owner_round_robin

    is_dist = dist.is_available() and dist.is_initialized()
    dp_world, group, dp_rank = _dp_world_and_group(dist)

    if not is_dist or dp_world <= 1:
        with _timer("compute"):
            with torch.no_grad():
                for pl in layers:
                    pl.merge_then_reinitialize(reinit_perm=reinit_perm)
        return

    owners = owner_round_robin(len(layers), dp_world)

    with _timer("compute"):
        with torch.no_grad():
            for i, pl in enumerate(layers):
                if owners[i] == dp_rank:
                    pl.merge_then_reinitialize(reinit_perm=reinit_perm)
                else:
                    pl.oft_R_in.data.zero_()
                    pl.oft_R_out.data.zero_()

    with _timer("comm"):
        for i, pl in enumerate(layers):
            src = torch.distributed.get_global_rank(group, owners[i])
            dist.broadcast(pl.weight.data, src=src, group=group)
            if reinit_perm:
                for buf in (pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv):
                    dist.broadcast(buf, src=src, group=group)
```

> `oft_R` is zeroed identically on every rank (owner via the merge, others explicitly), so it needs no broadcast. With `reinit_perm=False` perms are unchanged, so only `weight` crosses the wire — strictly less comm than mode A's 7 buffers.

- [ ] **Step 2: GPU validation — loss parity vs mode A**

Run (yours):
```bash
codexlog merge_sharded bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.merge_distributed=sharded
```
Expected: loss trajectory matches the mode-A baseline within noise; with profiling on, `compute` drops ≈`dp_world×` vs mode A.

- [ ] **Step 3: Commit**

```bash
git add src/patches/poet_merge_step.py
git commit -m "feat(poet): sharded compute + DP-group broadcast merge (Approach C)"
```

---

## Task 7: Validation sweep + CPU suite

Confirm correctness end-to-end and that the default path is untouched.

**Files:** none (validation only).

- [ ] **Step 1: Full CPU unit suite for touched areas**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_merge_dist.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py -v
```
Expected: all green (the 2 pre-existing unrelated failures noted in repo memory, if any, are out of scope).

- [ ] **Step 2: GPU bit-exactness — A vs B vs C**

Run all three modes for the same fixed seed/iters and compare a POET `weight` checksum + final loss. Because the merge is deterministic and pre-merge state is DP-identical, **A, B, and C must produce identical results** (B/C bit-identical to A in the `reinit_period=-1` steady state).

```bash
for mode in off comm_free sharded; do
  codexlog merge_parity_$mode bash scripts/train_poet_lie_orth.sh llama3 \
    optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
    optim.poet.merge_distributed=$mode training.train_iters=50
done
# compare final loss across the three logs in /lustre/home/zqiu/log/merge_parity_*.log
```
Expected: identical (or within fp-nondeterminism noise) loss across all three.

- [ ] **Step 3: Throughput confirmation**

Re-run the chosen mode with `POET_MERGE_PROFILE=1` and confirm `step_ms` dropped by ≈ the projected saving from the gate.

- [ ] **Step 4: Commit (docs/CHANGELOG only, if applicable)**

No code in this task. If a run log or note is worth keeping, commit it; otherwise nothing to commit.

---

## Self-review notes (filled by author)

- **Spec coverage:** §3 regimes → Tasks 5 (B), 6 (C), 4 (dispatch/default A). §4 measurement → Task 1 + GATE. §5 B design → Task 5 (incl. reanchor + reinit guard). §6 C design → Task 6 (two-phase, DP-group, `get_global_rank`). §7 flag plumbing → Task 3; ownership → Task 2. §8 validation → Task 7. §9 reinit/TP deferral → Task 5 guard + comments. All covered.
- **Open spec item (Kimi INT4 `pl.weight`):** explicitly *not* implemented here — it is a precondition to check before running B/C at Kimi scale, flagged in spec §9. No task claims to resolve it.
- **Type/name consistency:** `_poet_layers`, `_timer`, `_verify_dp_identical`, `_dp_world_and_group`, `_merge_rank0_broadcast`, `_merge_comm_free`, `_merge_sharded`, `owner_round_robin`, `owners_cost_aware` used consistently across tasks. `_merge_comm_free` signature `(layers, dist, iteration, reinit_perm, reanchor_period)` matches its call in Task 4.
