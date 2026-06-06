# Distributed (DP-sharded) Orthogonalization for LieOrthMomentum — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `LieOrthMomentum` shard the expensive Newton–Schulz orthogonalization across data-parallel ranks (round-robin) instead of every rank redundantly orthogonalizing the full `oft_R`, cutting that cost ≈`dp_world_size`× at scale — opt-in, numerically identical to the current replicated path.

**Architecture:** Today every DP rank holds identical `oft_R` (grads are DP-all-reduced by Megatron before `step()`) and redundantly runs the full orthogonalization. We split `step()`'s skew branch into three phases: (a) momentum EMA update — cheap, run on **all** ranks so `lie_m`/`lie_v` stay in sync; (b) a **pure** `_skew_update_buffer(dp_rank, dp_world)` that computes the generator `c·orthogonalize(−m)` **only for the round-robin-owned `oft_R` params** (lr applied at scatter, so the bf16 cast order matches the inline path bit-for-bit), zeros elsewhere, packed into one flat fp32 buffer; (c) one **bucketed `dist.all_reduce(SUM)`** of that buffer over the DP group, then scatter-apply to `oft_R`. Summing `[real_on_owner, 0_elsewhere]` is exact (adding zeros never perturbs bits) and `all_reduce` returns identical bits to every rank, so all ranks end with identical `oft_R` — **no drift, no shape constraints** (unlike the reference's per-chunk `all_gather`, which requires same-shape params). The replicated path is just this with `(dp_rank=0, dp_world=1)` and no collective, so it's behavior-preserving and the default.

**Why all_reduce-of-deltas, not the reference's all_gather:** `oft_R` params have heterogeneous shapes `(n_blocks_i, n_elems)` (`n_blocks` varies per layer). `muon_official.py`'s `dist.all_gather(params_pad[base_i:base_i+W], params_pad[base_i+rank])` requires every rank's contributed tensor in a chunk to be the **same shape**, which fails for POET. Flattening per-rank update deltas into one buffer and `all_reduce`-summing sidesteps shapes entirely and is exact (zeros).

**Tech Stack:** PyTorch (`torch 2.11`, `torch.distributed` gloo for CPU tests / NCCL at runtime), Megatron-LM `parallel_state` (DP group/rank/world), OmegaConf/Hydra, pytest.

**CPU test runner (base `python` lacks torch/omegaconf):**
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -v
```
Run all commands from the repo root `/lustre/fast/fast/zqiu/slm-research`.

> **Correctness invariant this relies on:** Megatron DP-all-reduces gradients into `main_grad` **before** `optimizer.step()`, so every DP rank sees **identical** `p.grad`. Therefore identical `oft_R` init + identical grads + identical (replicated) momentum updates ⇒ `oft_R` stays bit-identical across ranks every step. The sharded orthogonalization only changes *who computes which block*, then re-synchronizes — it must not change the result.

**Reference:** `/lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_official.py` (the round-robin + collective pattern). **Background:** [POET_dev.md](/lustre/fast/fast/zqiu/slm-research/POET_dev.md), [docs/muon_orthogonalizing_optimizer_poet.md](/lustre/fast/fast/zqiu/slm-research/docs/muon_orthogonalizing_optimizer_poet.md).

---

## Background: current step() and the plumbing chain

The optimizer is [src/optim/poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py) (`LieOrthMomentum`). Its `step()` skew branch currently, per skew param: updates `lie_m` (+`lie_v` if 2nd moment), then **inline** computes `gen = c·orthogonalize_skew_direction(−m)` and writes `p.add_(gen, alpha=lr)`. No `dist.*` anywhere.

A new optimizer knob flows through five layers (mirror the existing `lie_ortho_*` knobs):
1. **YAML** `configs/experiments/optim/poet_lie_orth.yaml` — `optim.poet.lie_ortho_distributed`.
2. **argv** `src/utils/megatron_args.py` (`kind=="poet"`) — `--poet-lie-ortho-distributed` (store_true).
3. **argparse** `launchers/pretrain_gpt_slm.py` (`add_slm_args`).
4. **config copy** `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`).
5. **builder** `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`) — resolve the DP context via `mpu` and pass it to `LieOrthMomentum`.

**Megatron DP API** (vendored, [parallel_state.py](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/parallel_state.py)): `get_data_parallel_group()`, `get_data_parallel_world_size()`, `get_data_parallel_rank()`. The lie builder already does `from megatron.core import parallel_state as mpu`.

**Data layout:** `oft_R` params are `(n_blocks, n_elems)`, `n_elems = b·(b−1)/2`. `n_blocks` (hence shape) varies per layer.

---

## File Structure

| File | Create / Modify | Responsibility |
|---|---|---|
| `src/optim/poet_lie_orth.py` | Modify | Split `step()` into momentum-update + pure `_skew_update_buffer(dp_rank, dp_world, active)` + scatter-apply; add `distributed`/`dp_*` ctor args + the gated `all_reduce`. |
| `src/optim/poet.py` | Modify | Builder resolves `mpu` DP ctx and passes it to `LieOrthMomentum` when `poet_lie_ortho_distributed`. |
| `launchers/pretrain_gpt_slm.py` | Modify | Add `--poet-lie-ortho-distributed` (store_true). |
| `src/utils/megatron_args.py` | Modify | Emit `--poet-lie-ortho-distributed` from `optim.poet.lie_ortho_distributed`. |
| `src/patches/poet_optimizer_setup.py` | Modify | Copy `args.poet_lie_ortho_distributed` → `config.poet_lie_ortho_distributed`. |
| `configs/experiments/optim/poet_lie_orth.yaml` | Modify | Add `lie_ortho_distributed: false` (documented knob). |
| `tests/unit/test_poet_lie_orth.py` | Modify | Refactor-equivalence + sharding-equivalence (pure, no dist) tests. |
| `tests/unit/test_poet_lie_orth_distributed.py` | Create | gloo 2-rank integration test: distributed `step()` == single-rank `step()`. |
| `tests/unit/test_pretrain_gpt_slm.py` | Modify | argparse flag test. |
| `tests/unit/test_megatron_args.py` | Modify | argv emission test. |
| `tests/unit/test_patch_poet_optimizer_setup.py` | Modify | config-copy test. |
| `POET_dev.md` | Modify | Note the distributed path (status, how to enable). |
| `CHANGELOG.md` | Modify | Log it. |

**Naming (use verbatim):** ctor kwargs `distributed: bool`, `dp_world_size: int`, `dp_rank: int`, `dp_group`; config/arg `poet_lie_ortho_distributed`; flag `--poet-lie-ortho-distributed`; YAML `lie_ortho_distributed`; methods `_skew_update_buffer`, `_apply_skew_update_buffer`.

---

## Task 1: Refactor `step()` into buffer-build + scatter-apply (behavior-preserving)

Split the skew branch so the per-param update is computed into a flat buffer and applied in a second pass, with `(dp_rank=0, dp_world=1)` reproducing today's result exactly. No distributed code yet.

**Files:**
- Modify: `src/optim/poet_lie_orth.py` (`step()`, and add `_skew_update_buffer` / `_apply_skew_update_buffer`)
- Test: `tests/unit/test_poet_lie_orth.py` (append)

- [ ] **Step 1: Write the failing test (helper contract: rank-0/world-1 owns every param)**

Append to `tests/unit/test_poet_lie_orth.py`. (Behavior-preservation of the refactor is guaranteed by the *existing* suite still passing — Step 5; this test pins the new helpers' contract.)

```python
def test_replicated_buffer_owns_all_params():
    # At (dp_rank=0, dp_world=1) every skew param is owned, so its buffer slice is
    # written (non-zero) — the replicated path covers everything.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum([dict(params=ps, use_skew=True, side="out", lr=0.1)], ortho_c=0.05)
    opt._lie_m_update(active=None)
    buf, slices = opt._skew_update_buffer(dp_rank=0, dp_world=1, active=None)
    assert len(slices) == 3
    for off, n, _, _ in slices:
        assert buf[off : off + n].abs().sum() > 0  # written, not left as zeros
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k replicated_buffer_owns_all -v`
Expected: FAIL — `AttributeError: 'LieOrthMomentum' object has no attribute '_lie_m_update'`.

- [ ] **Step 3: Refactor `step()` + add the two helpers**

In `src/optim/poet_lie_orth.py`, replace the entire `step` method (the `@torch.no_grad() def step(...)` block) with:

```python
    def _lie_m_update(self, active):
        """Phase (a): update lie_m (+ lie_v if used) for ALL skew params. Cheap; run on
        every rank so the momentum buffers stay in sync (grads are DP-identical)."""
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            b1, b2, v_mode = group["b1"], group["b2"], group["v_mode"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                g = g.float()
                st = self.state[p]
                if "lie_m" not in st:
                    st["lie_m"] = torch.zeros_like(g)
                    if self.ortho_use_second_moment:
                        if v_mode == "scalar":
                            st["lie_v"] = torch.zeros(g.shape[0], 1, dtype=g.dtype, device=g.device)
                        else:
                            st["lie_v"] = torch.zeros_like(g)
                st["lie_m"].mul_(b1).add_(g, alpha=1 - b1)
                if self.ortho_use_second_moment:
                    v = st["lie_v"]
                    if v_mode == "scalar":
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)

    def _iter_skew_params(self):
        """Deterministic, rank-identical ordering of skew params with a grad."""
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            for p in group["params"]:
                if p.grad is not None:
                    yield p, group

    def _skew_update_buffer(self, dp_rank, dp_world, active):
        """Phase (b), PURE (reads lie_m/lie_v, no mutation): compute the generator
        gen = ortho_c*orthogonalize(-dir) for the round-robin-OWNED skew params
        (i % dp_world == dp_rank), zeros for the rest, packed into one flat fp32 buffer.
        NOTE: lr is NOT folded in here — it is applied at scatter (alpha=lr) so the cast
        to bf16 happens in the same order as the inline path (gen.to(dtype) THEN *lr),
        making the buffer path bit-identical to the old inline update. (Folding lr in
        fp32 here would round differently in bf16 — verified ~3e-5 drift otherwise.)
        Returns (flat_buffer, slices=[(offset, numel, param, lr), ...])."""
        items = list(self._iter_skew_params())
        slices, total = [], 0
        for p, group in items:
            slices.append((total, p.numel(), p, group["lr"]))
            total += p.numel()
        if total == 0:
            return torch.zeros(0), []
        device = items[0][0].grad.device
        buf = torch.zeros(total, dtype=torch.float32, device=device)
        for i, (p, group) in enumerate(items):
            if (i % dp_world) != dp_rank:
                continue  # not this rank's block -> leave zeros (exact under all_reduce SUM)
            if self.alternating and group["side"] != active:
                continue  # inactive side -> no rotation written this step
            st = self.state[p]
            m = st["lie_m"]
            if self.ortho_use_second_moment:
                A_dir = -m / (st["lie_v"].sqrt() + group["eps"])
            else:
                A_dir = -m
            bsz = block_size_from_nelems(A_dir.shape[1])
            X = orthogonalize_skew_direction(
                vec_to_skew(A_dir, bsz),
                method=self.ortho_method,
                ns_steps=self.ortho_ns_steps,
            )
            gen = skew_to_vec(self.ortho_c * X, bsz)  # (n_blocks, n_elems) float; lr at scatter
            off, n = slices[i][0], slices[i][1]
            buf[off : off + n] = gen.reshape(-1)
        return buf, slices

    def _apply_skew_update_buffer(self, buf, slices):
        """Phase (d): scatter the (already all-reduced) flat buffer back onto oft_R,
        applying each param's lr. Cast order (gen.to(dtype) then alpha=lr) matches the
        inline path exactly, so the buffer/sharded path is bit-identical to replicated."""
        for off, n, p, lr in slices:
            p.add_(buf[off : off + n].view_as(p).to(p.dtype), alpha=lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = None
        if self.alternating:
            active = "out" if (self._alt_step // self.alternate_every) % 2 == 0 else "in"

        # --- skew branch: momentum (all ranks) -> owned-update buffer -> apply ---
        self._lie_m_update(active)
        dp_rank, dp_world = self._dp_rank, self._dp_world_size
        buf, slices = self._skew_update_buffer(dp_rank, dp_world, active)
        if self.distributed and dp_world > 1 and buf.numel() > 0:
            import torch.distributed as dist

            dist.all_reduce(buf, group=self.dp_group)
        self._apply_skew_update_buffer(buf, slices)

        # --- AdamW branch (non-skew params): unchanged, replicated ---
        for group in self.param_groups:
            if group["use_skew"]:
                continue
            beta1, beta2 = group["adamw_betas"]
            aeps, wd = group["adamw_eps"], group["adamw_wd"]
            lr = group["lr"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["moment1"] = torch.zeros_like(g)
                    st["moment2"] = torch.zeros_like(g)
                st["step"] += 1
                m1, m2 = st["moment1"], st["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)
                update = m1 / (aeps + m2.sqrt())
                bc1 = 1 - beta1 ** st["step"]
                bc2 = 1 - beta2 ** st["step"]
                scale = bc1 / bc2**0.5
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr / scale)

        if self.alternating:
            self._alt_step += 1
        return loss
```

- [ ] **Step 4: Add the new ctor fields (so `_dp_rank` etc. exist)**

In `LieOrthMomentum.__init__`, add these parameters to the signature (after `ortho_use_second_moment: bool = False,`, before `adamw_betas=(0.9, 0.95),`):

```python
        distributed: bool = False,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        dp_group=None,
```

and, immediately after `self.ortho_use_second_moment = bool(ortho_use_second_moment)`, add:

```python
        # DP-sharded orthogonalization (off by default = replicated path). When on and
        # dp_world_size > 1, each rank orthogonalizes only its round-robin slice of
        # oft_R, then one all_reduce(SUM) of the zero-padded update deltas re-syncs.
        self.distributed = bool(distributed)
        self._dp_world_size = int(dp_world_size)
        self._dp_rank = int(dp_rank)
        self.dp_group = dp_group
```

- [ ] **Step 5: Run the refactor test + the whole file to verify behavior is preserved**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: all pass — the new test plus every pre-existing `LieOrthMomentum` test (band, spectral, sign-flip, first-vs-second, momentum-persist, lie_v gating, AdamW branch) still green, proving the refactor changed nothing for `dp_world=1`.

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_lie_orth.py tests/unit/test_poet_lie_orth.py
git commit -m "refactor(poet): split LieOrthMomentum step into momentum/buffer/apply phases (behavior-preserving)"
```

---

## Task 2: Sharding equivalence — summed per-rank buffers == replicated buffer

Prove the round-robin sharding is correct **without** any real distributed runtime: simulate every rank in one process, sum their buffers (what `all_reduce(SUM)` does), and assert it equals the replicated `(rank=0, world=1)` buffer.

**Files:**
- Test: `tests/unit/test_poet_lie_orth.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_lie_orth.py`:

```python
@pytest.mark.parametrize("dp_world", [2, 3, 4])
def test_sharded_buffers_sum_to_replicated(dp_world):
    # Build several skew params of DIFFERENT shapes (heterogeneous n_blocks), one opt.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        b1=0.9, b2=0.95, eps=1e-8, ortho_c=0.05, ortho_method="muon", ortho_ns_steps=5,
    )
    opt._lie_m_update(active=None)  # momentum once (shared across the simulated ranks)
    replicated, _ = opt._skew_update_buffer(dp_rank=0, dp_world=1, active=None)
    summed = torch.zeros_like(replicated)
    for r in range(dp_world):
        buf_r, _ = opt._skew_update_buffer(dp_rank=r, dp_world=dp_world, active=None)
        summed += buf_r
    # Each param is owned by exactly one rank; zeros elsewhere ⇒ sum == replicated, exactly.
    assert torch.equal(summed, replicated), (summed - replicated).abs().max()


def test_sharded_owns_each_param_exactly_once():
    # Every skew param must be written by exactly one rank (no double-count, no drop).
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5, 4)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
    )
    opt._lie_m_update(active=None)
    dp_world = 3
    nonzero_owners = [0] * len(ps)
    for r in range(dp_world):
        buf, slices = opt._skew_update_buffer(dp_rank=r, dp_world=dp_world, active=None)
        for i, (off, n, _, _) in enumerate(slices):
            if buf[off : off + n].abs().sum() > 0:
                nonzero_owners[i] += 1
    assert all(c == 1 for c in nonzero_owners), nonzero_owners
```

- [ ] **Step 2: Run to verify it passes (sharding math already implemented in Task 1)**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k "sharded" -v`
Expected: PASS for all `dp_world` ∈ {2,3,4} and the ownership test. (These tests validate the Task-1 `_skew_update_buffer` round-robin; if they fail, the bug is in Task 1 — fix there.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_poet_lie_orth.py
git commit -m "test(poet): sharded update buffers sum to the replicated buffer (heterogeneous shapes)"
```

---

## Task 3: gloo 2-rank integration test — distributed step() == single-rank step()

Prove the real `dist.all_reduce` wiring end-to-end on CPU with 2 gloo ranks.

**Files:**
- Create: `tests/unit/test_poet_lie_orth_distributed.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_poet_lie_orth_distributed.py`:

```python
"""gloo 2-rank integration test for LieOrthMomentum's DP-sharded orthogonalization.
Each rank sees IDENTICAL grads (as Megatron guarantees post-all-reduce); the sharded
step must reproduce the single-rank (replicated) step bit-for-bit."""

import os

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn


def _single_rank_oft_R(seed):
    torch.manual_seed(seed)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    gs = [torch.randn_like(p) for p in ps]
    from src.optim.poet_lie_orth import LieOrthMomentum

    for p, g in zip(ps, gs):
        p.grad = g.clone()
    LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)], ortho_c=0.05
    ).step()
    return [p.data.clone() for p in ps]


def _worker(rank, world, q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)  # identical grads on every rank (DP invariant)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    for p in ps:
        p.grad = torch.randn_like(p)
    from src.optim.poet_lie_orth import LieOrthMomentum

    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        distributed=True,
        dp_world_size=world,
        dp_rank=rank,
        dp_group=dist.group.WORLD,
    )
    opt.step()
    if rank == 0:
        q.put([p.data.clone() for p in ps])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_distributed_step_matches_single_rank():
    ref = _single_rank_oft_R(seed=0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, q)) for r in range(2)]
    for p in procs:
        p.start()
    got = q.get(timeout=120)
    for p in procs:
        p.join(timeout=120)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
```

- [ ] **Step 2: Run the gloo test**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth_distributed.py -v`
Expected: PASS — the 2-rank gloo distributed step reproduces the single-rank `oft_R`. (If your CPU env blocks `spawn`/socket binds, this may be skipped/slow; the Task-2 pure-equivalence test is the primary guarantee. Bump `MASTER_PORT` if 29555 is taken.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_poet_lie_orth_distributed.py
git commit -m "test(poet): gloo 2-rank check that DP-sharded LieOrthMomentum equals single-rank"
```

---

## Task 4: argparse flag `--poet-lie-ortho-distributed`

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (`add_slm_args`, after the `--poet-lie-ortho-use-second-moment` line)
- Test: `tests/unit/test_pretrain_gpt_slm.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_lie_ortho_distributed():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    on = parser.parse_args(["--slm-config-path", "x.yaml", "--poet-lie-ortho-distributed"])
    off = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert on.poet_lie_ortho_distributed is True
    assert off.poet_lie_ortho_distributed is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho_distributed -v`
Expected: FAIL — `unrecognized arguments: --poet-lie-ortho-distributed`.

- [ ] **Step 3: Add the flag**

In `launchers/pretrain_gpt_slm.py`, immediately after the line `group.add_argument("--poet-lie-ortho-use-second-moment", action="store_true")`, insert:

```python
    # Shard the orthogonalization across data-parallel ranks (round-robin + all_reduce);
    # numerically identical to the replicated path. Off by default (dev/single-GPU).
    group.add_argument("--poet-lie-ortho-distributed", action="store_true")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho_distributed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py
git commit -m "feat(poet): add --poet-lie-ortho-distributed flag"
```

---

## Task 5: Emit the flag from the YAML (`megatron_args.py`)

**Files:**
- Modify: `src/utils/megatron_args.py` (`_optimizer_args`, the store_true block after the `lie_ortho_use_second_moment` append)
- Test: `tests/unit/test_megatron_args.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_emits_lie_ortho_distributed():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "q_optimizer": "lie_ortho", "lie_ortho_distributed": True})
    )
    assert "--poet-lie-ortho-distributed" in args


def test_poet_argv_omits_lie_ortho_distributed_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho-distributed" not in args
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_ortho_distributed -v`
Expected: FAIL — `--poet-lie-ortho-distributed` not in args.

- [ ] **Step 3: Add the store_true emission**

In `src/utils/megatron_args.py`, immediately after the block:

```python
        if poet.get("lie_ortho_use_second_moment", False):
            poet_args.append("--poet-lie-ortho-use-second-moment")
```

insert:

```python
        if poet.get("lie_ortho_distributed", False):
            poet_args.append("--poet-lie-ortho-distributed")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "lie_ortho_distributed or poet" -v`
Expected: new tests pass; existing poet tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit --poet-lie-ortho-distributed from optim.poet.lie_ortho_distributed"
```

---

## Task 6: Copy the flag into the optimizer config (`poet_optimizer_setup.py`)

**Files:**
- Modify: `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`, after the `poet_lie_ortho_use_second_moment` copy)
- Test: `tests/unit/test_patch_poet_optimizer_setup.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_poet_optimizer_setup.py`:

```python
def test_get_config_copies_lie_ortho_distributed(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    fake_training = types.SimpleNamespace()
    fake_training.get_megatron_optimizer_config = lambda args: (
        types.SimpleNamespace(optimizer="adam", lr=1e-3),
        {},
    )
    fake_training.get_megatron_optimizer = lambda config, model, **kwargs: "adam-optimizer"

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)

    patch_mod.apply()
    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=1,
        poet_scale=0.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_lie_ortho_distributed=True,
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_lie_ortho_distributed is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -k lie_ortho_distributed -v`
Expected: FAIL — `AttributeError: ... no attribute 'poet_lie_ortho_distributed'`.

- [ ] **Step 3: Add the config copy**

In `src/patches/poet_optimizer_setup.py`, in `_wrapped_get_config`, immediately after the line `config.poet_lie_ortho_use_second_moment = getattr(...)` (the multi-line `getattr` ending in `False\n        )`), insert:

```python
        config.poet_lie_ortho_distributed = getattr(args, "poet_lie_ortho_distributed", False)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_optimizer_setup.py tests/unit/test_patch_poet_optimizer_setup.py
git commit -m "feat(poet): copy poet_lie_ortho_distributed into the optimizer config"
```

---

## Task 7: Builder wires the DP context into `LieOrthMomentum`

When `poet_lie_ortho_distributed` is set, resolve the DP group/rank/world from `mpu` and pass them to `LieOrthMomentum`. Not CPU-unit-testable (needs Megatron handles); verify with `py_compile` + the downstream arg tests + the GPU validation (Task 8).

**Files:**
- Modify: `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`, the `LieOrthMomentum(...)` construction at ~line 606)

- [ ] **Step 1: Pass the DP context**

In `src/optim/poet.py`, replace the `LieOrthMomentum(...)` construction block (the `optimizer = LieOrthMomentum(param_groups, ortho_c=..., ..., **shared_kwargs)` call) with:

```python
        from megatron.core import parallel_state as mpu

        _lie_ortho_distributed = bool(getattr(config, "poet_lie_ortho_distributed", False))
        _dp_world = mpu.get_data_parallel_world_size() if _lie_ortho_distributed else 1
        _dp_rank = mpu.get_data_parallel_rank() if _lie_ortho_distributed else 0
        _dp_group = mpu.get_data_parallel_group() if _lie_ortho_distributed else None
        if _lie_ortho_distributed:
            logger.info(
                "[POET] Lie-orth DISTRIBUTED orthogonalization: dp_world=%s (round-robin + all_reduce)",
                _dp_world,
            )
        optimizer = LieOrthMomentum(
            param_groups,
            ortho_c=getattr(config, "poet_lie_ortho_c", 0.01),
            ortho_method=getattr(config, "poet_lie_ortho_method", "muon"),
            ortho_ns_steps=getattr(config, "poet_lie_ortho_ns_steps", 5),
            ortho_use_second_moment=getattr(config, "poet_lie_ortho_use_second_moment", False),
            distributed=_lie_ortho_distributed,
            dp_world_size=_dp_world,
            dp_rank=_dp_rank,
            dp_group=_dp_group,
            **shared_kwargs,
        )
```

- [ ] **Step 2: Verify the module compiles**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/optim/poet.py src/optim/poet_lie_orth.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/optim/poet.py
git commit -m "feat(poet): builder resolves DP context and enables sharded LieOrthMomentum"
```

---

## Task 8: YAML knob, full sweep, docs, and GPU validation

**Files:**
- Modify: `configs/experiments/optim/poet_lie_orth.yaml`, `POET_dev.md`, `CHANGELOG.md`

- [ ] **Step 1: Add the documented YAML knob**

In `configs/experiments/optim/poet_lie_orth.yaml`, immediately after the `lie_ortho_use_second_moment: false  # ...` line, insert:

```yaml
    lie_ortho_distributed: false # shard the NS orthogonalization across DP ranks (round-robin + all_reduce); identical result, ~dp_world× cheaper at scale. Off by default.
```

- [ ] **Step 2: Run the full POET unit sweep**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_lie_orth_distributed.py \
  tests/unit/test_poet_lie_momentum.py \
  tests/unit/test_patch_poet_merge.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patch_poet_optimizer_setup.py -v
```
Expected: all pass. Do not proceed past a red bar.

- [ ] **Step 3: Document in POET_dev.md**

In `POET_dev.md` §2.1, in the `lie_ortho` details block, append this sentence to the end of the **Status** paragraph:

```markdown
A DP-**sharded** orthogonalization path (`optim.poet.lie_ortho_distributed=true`, [poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py) `_skew_update_buffer`) round-robins the per-block Newton–Schulz across data-parallel ranks and re-syncs with one `all_reduce(SUM)` of the zero-padded update deltas — numerically identical to the replicated path (verified by a gloo 2-rank test), ≈`dp_world`× cheaper orthogonalization at scale. Off by default; a no-op at `dp_world=1`.
```

- [ ] **Step 4: Update CHANGELOG**

Add at the top of the current section of `CHANGELOG.md`:

```markdown
- feat(poet): DP-sharded orthogonalization for `LieOrthMomentum` (`optim.poet.lie_ortho_distributed=true`). Each data-parallel rank orthogonalizes only its round-robin slice of `oft_R`; one bucketed `all_reduce(SUM)` of zero-padded update deltas re-syncs (exact — adding zeros never perturbs bits, no shape constraints, unlike per-chunk all_gather). Numerically identical to the replicated path (gloo 2-rank test); ≈`dp_world`× cheaper NS at scale. Off by default, no-op at `dp_world=1`.
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth.yaml POET_dev.md CHANGELOG.md
git commit -m "docs(poet): lie_ortho_distributed knob + tracker/changelog notes"
```

- [ ] **Step 6: GPU validation — HAND OFF TO THE USER (do NOT run)**

The numerical-equivalence and wiring are unit-tested on CPU. The remaining check is a real 8-GPU run: confirm enabling the flag (a) trains identically and (b) speeds up. Provide and stop:

```
# A/B: same recipe, distributed off vs on — val curves should overlap.
codexlog lieorth_c8_repl  bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8
codexlog lieorth_c8_dist  bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true
```
Expected: the `[POET] Lie-orth DISTRIBUTED ... dp_world=8` log line appears; `val/loss` curves match `lieorth_c8_repl` (≈3.567) within run-to-run noise; per-step optimizer time for the skew branch drops. (At 60m the speedup is small — the NS is a minor fraction; the win grows with model size, the real target.)

---

## Self-Review: spec coverage

| Requirement | Covered by |
|---|---|
| Shard orthogonalization across GPUs (à la `muon_official`) | Task 1 (`_skew_update_buffer` round-robin owner) + Task 7 (DP ctx from `mpu`) |
| Re-sync so all ranks agree (the reference's all_gather) | Task 1 step() `all_reduce(SUM)` of zero-padded deltas — heterogeneous-shape-safe, exact |
| Numerically identical to replicated | Task 2 (sum==replicated, pure) + Task 3 (gloo 2-rank == single-rank) |
| Opt-in, off by default, no-op at `dp_world=1` | ctor `distributed=False`; step() guard `distributed and dp_world>1`; Tasks 4–8 plumbing default false |
| DP-only (no TP/PP/dist-optimizer) | builder already raises on those ([poet.py:552](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L552)); DP group via `mpu` |
| Don't regress the replicated path / other optimizers | Task 1 behavior-preserving (full lie_orth suite green); AdamW branch + `LieAlgebraMomentum` untouched |

**Placeholder scan:** none — every step has complete code + exact commands.

**Type/name consistency:** ctor `LieOrthMomentum(..., distributed, dp_world_size, dp_rank, dp_group)`; internals `self.distributed`, `self._dp_world_size`, `self._dp_rank`, `self.dp_group`; methods `_lie_m_update(active)`, `_iter_skew_params()`, `_skew_update_buffer(dp_rank, dp_world, active)`, `_apply_skew_update_buffer(buf, slices)`; config/arg `poet_lie_ortho_distributed`; argv `--poet-lie-ortho-distributed`; YAML `lie_ortho_distributed`. Builder→ctor mapping in Task 7 matches the signature added in Task 1 Step 4.

**Key risk & mitigation:** correctness hinges on grads being DP-identical at `step()` (Megatron all-reduces `main_grad` first) and on `oft_R` staying bit-identical across ranks (true: identical init + identical grads + `all_reduce`-exact updates). The `all_reduce(SUM)`-of-zeros is exact (no fp drift) and shape-agnostic — deliberately chosen over the reference's `all_gather`, which would break on POET's heterogeneous `oft_R` shapes. If a future change lets ranks diverge (e.g. per-rank RNG in the optimizer), this breaks — keep the optimizer's skew math deterministic given (grad, state).
