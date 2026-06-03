# poet0 — Single-Step POET Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a poet0 training variant that folds the POET block rotation into the weight every step (`merge_period=1`) while resampling the block permutation Ψ and resetting Adam momentum only on a slower cadence (`reinit_period=400`), so momentum stays coherent within each fixed-Ψ stretch.

**Architecture:** The legacy POET merge fuses three things at one cadence (`merge_period`): fold `R(Q)→W`, resample Ψ, reset momentum. This plan splits them onto two cadences. A new `reinit_period` config key controls Ψ-resample + momentum-reset; the fold runs every `merge_period`. A `reinit_perm` flag on `POETLinear.merge_then_reinitialize` gives a fold-only mode (keeps current Ψ), and the merge patch gates the momentum-reset half of `_reset_vanilla_oft_state` while always zeroing the master value (load-bearing for correctness).

**Tech Stack:** Python, OmegaConf/Hydra configs, Megatron-Core training patches, PyTorch, pytest. CPU unit tests via the repo CPU venv; GPU smoke run is the user's to launch.

**Spec:** [docs/superpowers/specs/2026-06-03-poet0-single-step-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet0-single-step-design.md)

**Conventions used throughout:**
- Repo root: `/lustre/fast/fast/zqiu/slm-research` (run all commands from here).
- CPU test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python` (the base `python` lacks omegaconf/torch — see project memory).
- **Script dry-run tests (Task 7 only) need the venv on `PATH`.** `scripts/train_*.sh` shell out to a bare `python -m launchers.train_megatron`; on the login node bare `python` lacks the repo deps, so the dry-run subprocess fails with `ModuleNotFoundError: omegaconf` *unless* the venv bin is first on `PATH`. Prefix those test commands with `PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH`. (Verified: the pre-existing `test_poet_script_supports_llama3` is red without this prefix and green with it — this is an environment quirk, not a code bug. Tasks 1-6 are pure in-process pytest and do **not** need it.)
- Commit style: single short conventional-commit sentence, anonymous (no co-author trailer). The repo has a pre-commit hook; let it run (do **not** pass `--no-verify`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/patches/poet_merge_step.py` | Cadence decision helper + per-step dispatch; split fold / Ψ-resample / momentum-reset | Modify |
| `src/utils/megatron_args.py` | Emit `--poet-reinit-period`; validate it's a multiple of `merge_period` | Modify |
| `launchers/pretrain_gpt_slm.py` | Register `--poet-reinit-period` argparse arg | Modify |
| `third_party/poet_torch/poet_layer.py` | `merge_then_reinitialize(reinit_perm=...)` fold-only mode | Modify |
| `configs/experiments/optim/poet0.yaml` | New experiment: `merge_period=1`, `reinit_period=400` | Create |
| `docs/experiments/poet0.md` | Experiment doc (required by pre-commit hook) | Create |
| `scripts/train_poet0.sh` | Launcher script pointing at `experiment=optim/poet0` | Create |
| `tests/unit/test_patch_poet_merge.py` | Tests for `_merge_decision` + `_reset_vanilla_oft_state` gating | Modify |
| `tests/unit/test_megatron_args.py` | Tests for `--poet-reinit-period` emission + validation | Modify |
| `tests/unit/test_pretrain_gpt_slm.py` | Test argparse accepts `--poet-reinit-period` | Modify |
| `tests/unit/test_poet_layers.py` | CPU test for fold-only `merge_then_reinitialize` | Modify |
| `tests/unit/test_train_scripts.py` | Dry-run smoke test for `train_poet0.sh` | Modify |

---

## Task 1: Cadence decision helper (`_merge_decision`)

A pure function that decides, per step, whether to fold and whether to also
reinit (resample Ψ + reset momentum). Pure python so it is unit-testable
without Megatron.

**Files:**
- Modify: `src/patches/poet_merge_step.py` (add module-level function near the top, after the imports / before `apply`)
- Test: `tests/unit/test_patch_poet_merge.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_patch_poet_merge.py`:

```python
def test_merge_decision_poet0_folds_every_step_reinits_every_400():
    from src.patches.poet_merge_step import _merge_decision

    # merge_period=1 (fold every step), reinit_period=400.
    assert _merge_decision(1, 1, 400) == (True, False)
    assert _merge_decision(399, 1, 400) == (True, False)
    assert _merge_decision(400, 1, 400) == (True, True)
    assert _merge_decision(800, 1, 400) == (True, True)


def test_merge_decision_legacy_folds_and_reinits_together():
    from src.patches.poet_merge_step import _merge_decision

    # merge_period=400, reinit_period=0 -> falls back to merge_period, so fold
    # and reinit always coincide (byte-identical to today's behavior).
    assert _merge_decision(200, 400, 0) == (False, False)
    assert _merge_decision(400, 400, 0) == (True, True)
    assert _merge_decision(800, 400, 0) == (True, True)


def test_merge_decision_disabled_or_iter_zero():
    from src.patches.poet_merge_step import _merge_decision

    assert _merge_decision(0, 1, 400) == (False, False)   # iteration 0 never merges
    assert _merge_decision(10, 0, 400) == (False, False)  # merge_period<=0 disables fold
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py -k merge_decision -v
```
Expected: FAIL — `ImportError: cannot import name '_merge_decision'`.

- [ ] **Step 3: Implement the helper**

In `src/patches/poet_merge_step.py`, after the `logger = logging.getLogger(__name__)`
line (around line 45) and before `@register_patch`, add:

```python
def _merge_decision(
    iteration: int, merge_period: int, reinit_period: int
) -> tuple[bool, bool]:
    """Decide, for ``iteration``, whether to fold and whether to also reinit.

    Returns ``(folding, reinit)``:

    * ``folding`` — fold ``R(Q)`` into ``W`` and reset ``Q`` this step (cadence
      ``merge_period``). poet0 sets ``merge_period=1`` → fold every step.
    * ``reinit`` — *additionally* resample the block permutation Ψ and reset Adam
      momentum (cadence ``reinit_period``; ``<=0`` falls back to ``merge_period``,
      reproducing the legacy fused behavior). A reinit can only happen on a step
      that also folds, so ``reinit_period`` should be a multiple of
      ``merge_period`` (validated at arg-build time in megatron_args).
    """
    if merge_period <= 0 or iteration <= 0 or iteration % merge_period != 0:
        return (False, False)
    gap = reinit_period if reinit_period > 0 else merge_period
    return (True, iteration % gap == 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py -k merge_decision -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/patches/poet_merge_step.py tests/unit/test_patch_poet_merge.py && \
git commit -F - <<'EOF'
feat(poet): add _merge_decision helper splitting fold from permute/momentum-reset cadence
EOF
```

---

## Task 2: Emit and validate `--poet-reinit-period` in megatron_args

**Files:**
- Modify: `src/utils/megatron_args.py` — the `kind == "poet"` branch of `_optimizer_args` ([line 250](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L250))
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py` (the `_poet_cfg` helper already exists there and defaults `merge_period=200`):

```python
def test_poet_argv_emits_reinit_period_zero_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-reinit-period" in args
    assert args[args.index("--poet-reinit-period") + 1] == "0"


def test_poet_argv_emits_reinit_period_when_set():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_size": 256, "merge_period": 1, "reinit_period": 400})
    )
    assert args[args.index("--poet-merge-period") + 1] == "1"
    assert args[args.index("--poet-reinit-period") + 1] == "400"


def test_poet_argv_rejects_reinit_period_not_multiple_of_merge():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="multiple of"):
        _optimizer_args(
            _poet_cfg({"block_size": 256, "merge_period": 3, "reinit_period": 400})
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k reinit_period -v
```
Expected: FAIL — `--poet-reinit-period` not in args / no ValueError raised.

- [ ] **Step 3: Implement emission + validation**

In `src/utils/megatron_args.py`, in the `if kind == "poet":` branch, right after
`poet = optim.poet` ([line 251](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L251)), add the validation:

```python
        poet = optim.poet
        reinit_period = int(poet.get("reinit_period", 0))
        merge_period = int(poet.merge_period)
        if reinit_period > 0 and merge_period > 0 and reinit_period % merge_period != 0:
            raise ValueError(
                f"poet.reinit_period ({reinit_period}) must be a multiple of "
                f"poet.merge_period ({merge_period}); reinit boundaries must land "
                "on a folding step."
            )
```

Then in the `poet_args` list, immediately after the `"--poet-merge-period", poet.merge_period,`
pair ([lines 269-270](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L269)), add:

```python
            "--poet-reinit-period",
            reinit_period,
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k "reinit_period or poet" -v
```
Expected: PASS for the three new tests (and the pre-existing block-size/count poet tests still pass). Note: a pre-existing failure asserting `--poet-merge-period == "200"` for `experiment=optim/poet` may already be red independent of this change — do not "fix" it here.

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py && \
git commit -F - <<'EOF'
feat(poet): emit --poet-reinit-period and validate it divides merge_period
EOF
```

---

## Task 3: Register `--poet-reinit-period` in the launcher

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` — `add_slm_args` ([line 45](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L45))
- Test: `tests/unit/test_pretrain_gpt_slm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_poet_reinit_period():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        ["--poet", "--poet-merge-period", "1", "--poet-reinit-period", "400"]
    )
    assert args.poet_merge_period == 1
    assert args.poet_reinit_period == 400


def test_add_slm_args_poet_reinit_period_defaults_zero():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--poet"])
    assert args.poet_reinit_period == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py -k reinit_period -v
```
Expected: FAIL — `unrecognized arguments: --poet-reinit-period`.

- [ ] **Step 3: Register the arg**

In `launchers/pretrain_gpt_slm.py`, immediately after the `--poet-merge-period`
line ([line 45](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L45)), add:

```python
    # Cadence (optimizer steps) at which the block permutation is resampled AND
    # Adam momentum is reset. 0 = fall back to --poet-merge-period (legacy: fold,
    # resample, reset all fire together). poet0 sets merge_period=1 (fold each
    # step) + reinit_period=400 so Ψ and momentum stay coherent for 400-step
    # stretches while Q is folded into W every step.
    group.add_argument("--poet-reinit-period", type=int, default=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py -k reinit_period -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py && \
git commit -F - <<'EOF'
feat(poet): register --poet-reinit-period launcher arg
EOF
```

---

## Task 4: Fold-only mode for `POETLinear.merge_then_reinitialize`

Add a `reinit_perm: bool = True` parameter. `True` keeps today's behavior
(resample Ψ + re-permute weight + update perm buffers). `False` folds with the
**current** Ψ, re-permutes the folded weight back into the current layout using
the existing inverse perms, and leaves Ψ buffers untouched. CPU-testable via the
`exp` parameterization (`torch.linalg.matrix_exp`; `block_diag_lr_matmul_decoupled`
is pure torch).

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` — `POETLinear.merge_then_reinitialize` ([line 682](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L682))
- Test: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_layers.py`:

The key correctness property of *any* merge is **forward-invariance**: folding
`R(oft_R)` into `W` and zeroing `oft_R` moves the rotation but must NOT change the
layer's output. A wrong fold-only re-permutation would silently break this, so we
assert it directly (verified on CPU with `exp`: legacy merge diff ~1.8e-7).

```python
def test_merge_fold_only_is_forward_invariant_and_keeps_perm():
    import torch
    from poet_torch import POETLinear

    torch.manual_seed(0)
    layer = POETLinear(
        in_features=8, out_features=8, block_count=1,
        dtype=torch.float32, parameterization="exp",
    )
    layer.random_init_parameters()  # nonzero oft_R so the fold actually changes W
    x = torch.randn(4, 8, dtype=torch.float32)

    out_before = layer(x).detach().clone()
    perm_in_before = layer.perm_in.clone()
    perm_out_before = layer.perm_out.clone()
    weight_before = layer.weight.clone()

    layer.merge_then_reinitialize(reinit_perm=False)

    # Forward output unchanged (rotation moved into W, not lost) ...
    assert torch.allclose(out_before, layer(x), atol=1e-4)
    # ... Ψ unchanged (fold-only) ... oft_R reset ... weight absorbed the rotation.
    assert torch.equal(layer.perm_in, perm_in_before)
    assert torch.equal(layer.perm_out, perm_out_before)
    assert torch.count_nonzero(layer.oft_R_in) == 0
    assert torch.count_nonzero(layer.oft_R_out) == 0
    assert not torch.allclose(layer.weight, weight_before)


def test_merge_reinit_perm_true_is_forward_invariant_and_resamples_perm():
    import torch
    from poet_torch import POETLinear

    torch.manual_seed(0)
    layer = POETLinear(
        in_features=8, out_features=8, block_count=1,
        dtype=torch.float32, parameterization="exp",
    )
    layer.random_init_parameters()
    x = torch.randn(4, 8, dtype=torch.float32)

    out_before = layer(x).detach().clone()
    perm_in_before = layer.perm_in.clone()

    layer.merge_then_reinitialize(reinit_perm=True)

    # Still forward-invariant (weight re-permuted to match the new Ψ) ...
    assert torch.allclose(out_before, layer(x), atol=1e-4)
    # ... but Ψ WAS resampled (collision prob for 8! is negligible).
    assert not torch.equal(layer.perm_in, perm_in_before)
    assert torch.count_nonzero(layer.oft_R_in) == 0


def test_merge_then_reinitialize_defaults_to_reinit():
    import inspect
    from poet_torch import POETLinear

    sig = inspect.signature(POETLinear.merge_then_reinitialize)
    assert sig.parameters["reinit_perm"].default is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_layers.py -k "fold_only or reinit_perm or defaults_to_reinit" -v
```
Expected: FAIL — `merge_then_reinitialize() got an unexpected keyword argument 'reinit_perm'`.

- [ ] **Step 3: Implement fold-only mode**

Replace the body of `POETLinear.merge_then_reinitialize`
([lines 682-712](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L682)). The fold (computing `expected` in natural order) is unchanged;
only the permutation handling branches:

```python
    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        # Same math as POETLinear.merge_then_reinitialize, but float compute + requantize
        R_out, R_in = self._merge_R()

        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul_decoupled(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        if reinit_perm:
            # Legacy: generate a NEW permutation, re-permute the folded weight into
            # the new block-aligned layout, and update the perm buffers.
            device = self.weight.device
            perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
            perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
            perm_in_inv = torch.argsort(perm_in).to(torch.int32)
            perm_out_inv = torch.argsort(perm_out).to(torch.int32)

            expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
            self.weight.detach().copy_(expected)

            self.perm_in.copy_(perm_in)
            self.perm_in_inv.copy_(perm_in_inv)
            self.perm_out.copy_(perm_out)
            self.perm_out_inv.copy_(perm_out_inv)
        else:
            # Fold-only (poet0 non-boundary step): keep the CURRENT permutation.
            # Re-permute the folded weight back into the current block-aligned
            # layout using the existing inverse perms so the next forward with the
            # unchanged Ψ is consistent. Perm buffers are left untouched.
            expected = expected.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
            self.weight.detach().copy_(expected)

        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_layers.py -k "fold_only or reinit_perm or defaults_to_reinit" -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Confirm quantized siblings are not on poet0's path**

poet0 uses `block_count=1`, `parameterization=cayley`, `use_poet_adam=false` →
the non-quantized float `POETLinear` (line 509). The quantized
`merge_then_reinitialize` siblings (lines 781 `POETLinearNeurips`, 949
`QPOETLinear`) are only reached by int8/4bit paths, which poet0 does not enable.
Verify nothing in the poet0 config selects them:

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
grep -n "QPOETLinear\|POETLinearNeurips\|prepare_model_for_int8" src/optim/poet_layers.py || echo "no quantized POET layer wiring in poet_layers.py — OK"
```
Expected: confirms `src/optim/poet_layers.py` instantiates the float `POETLinear`
only (no int8/4bit wrapper). If a quantized sibling IS reachable, thread the same
`reinit_perm` parameter into it before proceeding.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add third_party/poet_torch/poet_layer.py tests/unit/test_poet_layers.py && \
git commit -F - <<'EOF'
feat(poet): add fold-only mode (reinit_perm=False) to POETLinear.merge_then_reinitialize
EOF
```

---

## Task 5: Wire the split cadences into the merge patch

Thread `reinit_perm` through `_run_merge`, gate the momentum-reset half of
`_reset_vanilla_oft_state` behind `reset_moments` (always zero the master
**value**), and rewrite `_wrapped` to dispatch via `_merge_decision`.

**Files:**
- Modify: `src/patches/poet_merge_step.py`
- Test: `tests/unit/test_patch_poet_merge.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_patch_poet_merge.py`:

```python
def _make_reset_fixture():
    """Model with one oft_R param + a fake Megatron optimizer holding a separate
    fp32 master with nonzero Adam moments. Mirrors _iter_model_master_pairs'
    plain-Float16 layout (float16_groups / fp32_from_float16_groups)."""
    import torch
    import torch.nn as nn

    model = nn.Module()
    model.oft_R_in = nn.Parameter(torch.ones(4))  # the bf16 model tensor
    master = nn.Parameter(torch.full((4,), 3.0))   # separate fp32 master, nonzero
    torch_opt = torch.optim.Adam([master], lr=1e-3)
    torch_opt.state[master] = {
        "exp_avg": torch.ones(4),
        "exp_avg_sq": torch.ones(4),
        "step": torch.tensor(5.0),
    }

    class _FakeInner:
        def __init__(self):
            self.float16_groups = [[model.oft_R_in]]
            self.fp32_from_float16_groups = [[master]]
            self.optimizer = torch_opt

    return model, _FakeInner(), torch_opt, master


def test_reset_vanilla_oft_state_keeps_moments_when_reset_moments_false():
    import torch

    from src.patches.poet_merge_step import _reset_vanilla_oft_state

    model, opt, torch_opt, master = _make_reset_fixture()
    _reset_vanilla_oft_state(opt, model, iteration=5, reset_moments=False)

    # Master VALUE always zeroed (prevents spring-back) ...
    assert torch.count_nonzero(master.data) == 0
    # ... but momentum + step preserved (poet0 persists momentum).
    assert torch.count_nonzero(torch_opt.state[master]["exp_avg"]) == 4
    assert torch_opt.state[master]["step"].item() == 5.0


def test_reset_vanilla_oft_state_zeros_moments_when_reset_moments_true():
    import torch

    from src.patches.poet_merge_step import _reset_vanilla_oft_state

    model, opt, torch_opt, master = _make_reset_fixture()
    _reset_vanilla_oft_state(opt, model, iteration=400, reset_moments=True)

    assert torch.count_nonzero(master.data) == 0
    assert torch.count_nonzero(torch_opt.state[master]["exp_avg"]) == 0
    assert torch_opt.state[master]["step"].item() == 0.0


def test_run_merge_forwards_reinit_perm_false_keeps_perm():
    import torch
    import torch.nn as nn
    from poet_torch import POETLinear

    from src.optim.poet_layers import POETMegatronLinear
    from src.patches.poet_merge_step import _run_merge

    torch.manual_seed(0)
    pl = POETLinear(
        in_features=8, out_features=8, block_count=1,
        dtype=torch.float32, parameterization="exp",
    )
    pl.random_init_parameters()
    wrapper = POETMegatronLinear(pl)
    model = nn.Module()
    model.layer = wrapper

    perm_in_before = pl.perm_in.clone()

    class _FakeDist:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def is_initialized():
            return False

    _run_merge(model, _FakeDist(), iteration=5, reinit_perm=False)

    assert torch.equal(pl.perm_in, perm_in_before)
    assert torch.count_nonzero(pl.oft_R_in) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py -k "reset_vanilla or run_merge_forwards" -v
```
Expected: FAIL — `_reset_vanilla_oft_state()`/`_run_merge()` got an unexpected keyword argument.

- [ ] **Step 3: Add `reset_moments` to `_reset_vanilla_oft_state`**

In `src/patches/poet_merge_step.py`, change the signature
([line 128](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L128)):

```python
def _reset_vanilla_oft_state(optimizer, model, iteration: int, reset_moments: bool = True) -> None:
```

In the per-pair loop, gate **only** the moment-zeroing (keep the value-zero
unconditional). Replace lines 186-189:

```python
            # Zero the fp32 master VALUE (no-op if master IS the model tensor,
            # which the merge already zeroed). ALWAYS done — load-bearing against
            # spring-back of the just-merged rotation.
            if master_p is not model_p:
                master_p.detach().zero_()
            n_val += 1
            # Moments reset only when reinit fires (Ψ changed -> new coordinate
            # frame). poet0 non-boundary steps keep momentum (reset_moments=False).
            if reset_moments:
                _zero_moments(master_p, torch_opt)
```

Wrap the per-group `step` reset loop (lines 195-206) in `if reset_moments:`:

```python
    n_groups = 0
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
                n_groups += 1
```

Update the log line (lines 208-213) to record the mode:

```python
    logger.info(
        "[POET] oft_R reset at iter %d: zeroed value for %d masters; moments %s (%d group-steps)",
        iteration,
        n_val,
        "reset" if reset_moments else "kept",
        n_groups,
    )
```

- [ ] **Step 4: Add `reinit_perm` to `_run_merge`**

Change the signature ([line 216](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L216)):

```python
def _run_merge(model, dist, iteration: int, reinit_perm: bool = True) -> None:
```

and the merge call ([line 235](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L235)):

```python
                if rank == 0:
                    pl.merge_then_reinitialize(reinit_perm=reinit_perm)
```

- [ ] **Step 5: Rewrite `_wrapped` to dispatch via `_merge_decision`**

Replace the body of `_wrapped` ([lines 56-84](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L56)):

```python
    def _wrapped(*args, **kwargs):
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
        _run_merge(model, dist, iteration, reinit_perm=do_reinit)
        # Megatron-Adam path (default): reset momentum ONLY when Ψ is resampled
        # (do_reinit) — otherwise momentum persists across the per-step fold. The
        # master VALUE is zeroed every fold regardless (inside _reset_vanilla_oft_state).
        if not getattr(opts, "poet_use_poet_adam", False):
            optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
            if optimizer is not None:
                _reset_vanilla_oft_state(optimizer, model, iteration, reset_moments=do_reinit)
        return ret
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py -v
```
Expected: PASS — the three new tests plus the pre-existing registration/cache tests.

- [ ] **Step 7: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/patches/poet_merge_step.py tests/unit/test_patch_poet_merge.py && \
git commit -F - <<'EOF'
feat(poet): split fold from permute/momentum-reset in merge patch (keep master-value zero unconditional)
EOF
```

---

## Task 6: poet0 experiment config + doc

**Files:**
- Create: `configs/experiments/optim/poet0.yaml`
- Create: `docs/experiments/poet0.md` (required by the pre-commit hook "Every experiment YAML has a matching docs/experiments/<name>.md")
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet0_experiment_yaml_sets_single_step_cadences():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet0.yaml")
    assert cfg.experiment.name == "poet0"
    assert cfg.optim.poet.merge_period == 1
    assert cfg.optim.poet.reinit_period == 400
    # poet0 keeps the stock optimizer (no Pion imports yet).
    assert cfg.optim.poet.use_poet_adam is False
    assert cfg.optim.poet.parameterization == "cayley"
    assert cfg.optim.poet.q_optimizer == "adam"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k poet0_experiment -v
```
Expected: FAIL — file `configs/experiments/optim/poet0.yaml` does not exist.

- [ ] **Step 3: Create the experiment config**

Create `configs/experiments/optim/poet0.yaml`:

```yaml
# @package _global_
# poet0: Single-Step POET.
#
# Baseline of the POET-X x Pion pipeline (docs/poetx_pion_pipeline.md S1): the
# block rotation Q is folded into the live weight W EVERY step (merge_period=1,
# born at identity, small angle, merged, reset), while the block permutation Psi
# is resampled AND Adam momentum is reset only every reinit_period steps. Between
# reinit boundaries the optimizer momentum persists in one coherent coordinate
# frame; at a boundary Psi changes (for full neuron-pair coverage) and momentum
# resets with it. Imports NONE of the Pion geometry (tangent grad, scalar-v Lie
# momentum, RMS-alpha, low-order Cayley, alternating) — those are later ablations.
experiment:
  name: poet0
  family: optim
  description: |
    Single-Step POET (merge_period=1, reinit_period=400). Folds R(oft_R) into
    the base weight every step and resets oft_R to identity, keeping Adam
    momentum across folds; resamples the block permutation and resets momentum
    on the slower reinit cadence. Same stock Megatron-Adam path, k=3 Cayley,
    two-sided rotations, and architectural unfusing as experiment=optim/poet —
    only the two cadences differ. Hypothesis: born-at-identity per-step folds
    are stable and train at least as well as the merge_period=400 baseline.
  references:
    - "POET"
  patches:
    - model_unfuse_linears    # unfuse fused qkv/fc1 at build time (pre-DDP)
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
    - training_log_eta        # prepend "ETA: HhMMm" to the per-iteration log
    - wandb_metric_normalize  # canonicalize W&B metric keys + add tokens_seen / step_time
  required_capabilities: []

optim:
  type: poet
  lr: 1.0e-3
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  poet:
    block_count: 1
    cache_mode: none
    init_type: normalized
    mup_alpha: 1.0
    merge_period: 1         # fold R(Q) into W every step (single-step)
    reinit_period: 400      # resample Psi + reset Adam momentum every N steps
    scale: 0.5
    use_poet_adam: false
    parameterization: cayley
    q_optimizer: adam
    muon_theta: 0.1
    muon_ns_steps: 5
    muon_momentum: 0.95
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 4: Create the experiment doc**

Create `docs/experiments/poet0.md`:

```markdown
# poet0 — Single-Step POET

Baseline of the POET-X × Pion pipeline
([docs/poetx_pion_pipeline.md](../poetx_pion_pipeline.md) §1). Same stack as
[`experiment=optim/poet`](./poet.md), with two cadences split apart:

- **`merge_period: 1`** — every step, fold `R(oft_R)` into the base weight and
  reset `oft_R` to identity. The per-step rotation angle stays small.
- **`reinit_period: 400`** — every 400 steps, *also* resample the block
  permutation Ψ and reset Adam momentum. Between boundaries the momentum
  persists in one coherent coordinate frame; the `oft_R` master **value** is
  zeroed on every fold regardless (prevents the just-merged rotation from
  springing back on the next optimizer step).

Everything else matches `optim/poet`: stock Megatron-Adam on `oft_R`
(`use_poet_adam: false`), k=3 Cayley (`parameterization: cayley`), two-sided
rotations, `scale: 0.5`, and qkv/fc1 unfusing. `reinit_period` must be a
multiple of `merge_period` (validated at arg-build time).

Run with [`scripts/train_poet0.sh`](../../scripts/train_poet0.sh) (60m dev
scale by default) or `experiment=optim/poet0` on any launcher.

**Out of scope** (later ablations, layered on this baseline): tangent-space
gradient, scalar-`v` Lie momentum, RMS-α step size, low-order Cayley,
alternating single-sided update, sharded merge.
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k poet0_experiment -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add configs/experiments/optim/poet0.yaml docs/experiments/poet0.md tests/unit/test_megatron_args.py && \
git commit -F - <<'EOF'
feat(poet): add poet0 single-step experiment config + doc
EOF
```

---

## Task 7: poet0 training script

**Files:**
- Create: `scripts/train_poet0.sh`
- Test: `tests/unit/test_train_scripts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_train_scripts.py`:

```python
def test_poet0_script_supports_llama3():
    proc = _run("train_poet0.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout
    assert "poet" in proc.stdout
    assert "--poet-merge-period" in proc.stdout
    assert "--poet-reinit-period" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run (note the `PATH` prefix — see Conventions):
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k poet0 -v
```
Expected: FAIL — `bash: scripts/train_poet0.sh: No such file or directory` (subprocess `check=True` raises `CalledProcessError`).

- [ ] **Step 3: Create the script**

Create `scripts/train_poet0.sh` as a clone of
[scripts/train_poet_dev.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_dev.sh)
with `experiment=optim/poet` → `experiment=optim/poet0` and an updated header.
The fastest reliable way is to copy and substitute, then fix the header comment:

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
sed 's#experiment=optim/poet#experiment=optim/poet0#' \
    scripts/train_poet_dev.sh > scripts/train_poet0.sh && \
chmod +x scripts/train_poet0.sh
```

Then replace the top comment block (lines 4-8) of `scripts/train_poet0.sh` so it
documents poet0 rather than the generic dev variant. Open the file and set the
header to:

```bash
# poet0 variant: same harness as train_poet_dev.sh (tiny 60m dev scale,
# seq_length=256, ablation_40x, cosine_poet, untied embeddings), but uses
# experiment=optim/poet0 — Single-Step POET: fold the block rotation into W
# every step (merge_period=1) and resample the permutation + reset Adam momentum
# every reinit_period (400) steps. Any "$@" override still wins.
```

Verify the only functional difference from the dev script is the experiment:

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
diff <(grep -v '^#' scripts/train_poet_dev.sh) <(grep -v '^#' scripts/train_poet0.sh)
```
Expected: a single hunk changing `experiment=optim/poet` to `experiment=optim/poet0`.

- [ ] **Step 4: Run test to verify it passes**

Run (note the `PATH` prefix — see Conventions):
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k poet0 -v
```
Expected: PASS. (The dry-run resolves `experiment=optim/poet0`, which now emits
both `--poet-merge-period 1` and `--poet-reinit-period 400`.)

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add scripts/train_poet0.sh tests/unit/test_train_scripts.py && \
git commit -F - <<'EOF'
feat(poet): add train_poet0.sh single-step launcher script
EOF
```

---

## Final verification (after all tasks)

- [ ] **Run the in-process unit tests (Tasks 1-6):**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_poet_layers.py -v
```
Expected: all new tests PASS. Pre-existing reds (e.g. the `--poet-merge-period == "200"` assertion against `experiment=optim/poet`, noted in project memory) are unrelated to this change — confirm they were red *before* this work and leave them.

- [ ] **Run the script dry-run test (Task 7) with the venv on `PATH`:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k poet -v
```
Expected: `test_poet0_script_supports_llama3` and the pre-existing
`test_poet_script_supports_llama3` both PASS (the `PATH` prefix lets the script's
bare `python` resolve to the venv — without it, both are red on the login node).

- [ ] **Static checks on edited Python:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile \
  src/patches/poet_merge_step.py src/utils/megatron_args.py \
  launchers/pretrain_gpt_slm.py third_party/poet_torch/poet_layer.py && \
ruff check src/patches/poet_merge_step.py src/utils/megatron_args.py \
  launchers/pretrain_gpt_slm.py 2>/dev/null || echo "ruff not configured for third_party — OK"
```

- [ ] **GPU smoke run (USER — do not run from the agent):** launch the 60m dev
  recipe and confirm (a) no per-step loss spike, (b) the merge log shows
  "moments kept" on non-boundary steps and "moments reset" + a new Ψ at the 400
  boundary, (c) loss tracks or beats the `merge_period=400` baseline:

```bash
codexlog poet0_smoke bash scripts/train_poet0.sh llama3 training.train_iters=50 optim.poet.reinit_period=20
```
(Short `reinit_period=20` makes a boundary land inside a 50-step smoke so both
branches are exercised. The user runs this on the cluster.)

---

## Self-Review Notes (author)

- **Spec coverage:** §4.1→Task 3, §4.2→Task 2, §4.3 (fold-only)→Task 4, §4.4
  (cadence split + reset gating)→Tasks 1 & 5, §4.5→Task 6, §4.6→Task 7, §3
  multiple-of validation→Task 2, §6 tests→Tasks 1-7. All covered.
- **Master-value zero is unconditional** in Task 5 Step 3 — matches spec §2(a)/§3.
- **Backward compat:** `reinit_period` defaults to 0 everywhere (Tasks 2,3) →
  `_merge_decision` falls back to `merge_period` (Task 1) → legacy poet
  unchanged. Verified by `test_merge_decision_legacy_*`.
- **Naming consistency:** `_merge_decision`, `reinit_perm`, `reset_moments`,
  `--poet-reinit-period`, `reinit_period` used identically across all tasks.
```
