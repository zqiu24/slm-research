# Alternating-on-POETX (integrated, both-momenta) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the POET champion behavior — write one rotation side per step while keeping **both** Lie momenta fresh (`lie_ortho` + `lie_alternating`, val/loss ≈3.5332) — onto the POETX forward-frame layer (`POETXLinear`), and add the one both-momenta-compatible POETX speedup: an **active-only merge fold** that skips the frozen side's identity Cayley.

**Architecture:** Two phases. **Phase 1** is config-only (zero `src/` changes): a champion recipe with `single_step_x=true` + `lie_alternating=true` builds a plain `POETXLinear` whose merge folds *both* sides (the frozen side's `oft_R=0` → `R=I` → no-op), reproducing the champion at POETX forward speed. **Phase 2** integrates `alternating` into `POETXLinear` so the merge folds **only the active side**; forward/backward stay both-sides (`POETXSingleStepFunction` returns both `grad_oft_R_in` and `grad_oft_R_out`, so both momenta stay fed — the load-bearing ingredient), and the optimizer's *write* + the merge's *fold* read the same `alt_state` iteration so they always target the same side.

**Tech Stack:** PyTorch (`torch.autograd.Function`), Megatron-LM training hooks (patched), Hydra/OmegaConf configs, pytest (CPU unit tests). Python 3.12 via the repo test venv.

---

## Conventions used by every task

**CPU test runner** (base `python` lacks torch/omegaconf; this venv has poet_torch editable-installed):

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
cd /lustre/fast/fast/zqiu/slm-research
```

Run a single test: `$PY -m pytest tests/unit/<file>::<test> -v`
(A harmless `CUDA driver ... too old` UserWarning prints on this CPU node — tests still run on CPU. Ignore it.)

**GPU acceptance runs are the user's to launch** — every task that needs a GPU run ends by printing the exact command and stopping. Never launch a GPU run.

---

## File Structure

| File | Responsibility | Phase |
|---|---|---|
| `configs/experiments/optim/poet_lie_orth_alt.yaml` (new) | Champion recipe: `single_step_x` + `lie_alternating`, head-off, lr 3e-3, c=8, distributed, `single_step_x_alternating=false` | 1 |
| `docs/experiments/poet_lie_orth_alt.md` (new) | Experiment doc (design/plan/baseline/expectation links) | 1 |
| `scripts/train_poet_lie_orth_alt.sh` (new) | Launcher (mirror of `train_poet_lie_orth_alt_x.sh`, experiment swapped) | 1 |
| `third_party/poet_torch/poetx_layer.py` | `POETXLinear` gains `alternating`/`alternate_every`; hosts `_fold_active_side`; `AlternatingPOETXLinear` becomes a thin subclass setting `alternating=True` | 2 |
| `src/optim/poet_lie_orth.py` | `_active_side`: the `alternating` mode reads `alt_state` (not the internal `_alt_step` counter) | 2 |
| `src/optim/poet_layers.py` | walk builds `POETXLinear(alternating=True)` for `single_step_x` + `lie_alternating` | 2 |
| `src/patches/poet_apply_to_model.py` | thread `lie_alternating` into the walk | 2 |
| `src/patches/poet_merge_step.py` | `_merge_layers`: route the active-only fold by the `alternating` flag, not by `isinstance` | 2 |
| `tests/unit/test_megatron_args.py` | accept `single_step_x` + `lie_alternating`; new config loads | 1 |
| `tests/unit/test_poetx_layer.py` | `_fold_active_side` both-sides parity (closes the Task-7 `"out"` gap), fp64 | 2 |
| `tests/unit/test_poet_lie_orth.py` | `alternating` mode writes the `alt_state` side (update existing test) | 2 |
| `tests/unit/test_poet_layers.py` | walk-selection: `single_step_x` + `lie_alternating` → `POETXLinear(alternating=True)` | 2 |
| `tests/unit/test_alternating_poetx.py` | merge routes `POETXLinear(alternating=True)` to active-only fold; write==fold consistency | 2 |

**Intermediate-state guarantee:** between Phase-2 tasks every state stays correct. After Task 5 the walk builds `POETXLinear(alternating=True)`, but until Task 6 the merge driver still folds it *both-sides* (the frozen side is identity → no-op), i.e. Phase-1 behavior. Task 6 flips it to active-only. No task leaves a broken merge.

---

## Background facts (verified against the code, 2026-06-08)

- `POETXSingleStepFunction.backward` returns **both** `grad_oft_R_in` and `grad_oft_R_out` ([poetx_ops.py:65-67](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py#L65-L67)) → both momenta stay fed.
- The optimizer's `alternating` mode gates the *write* by the param group's `side` ([poet_lie_orth.py:91-98](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L91-L98)) and currently reads the internal `_alt_step` counter — this is what Task 4 changes.
- `_lie_m_update` only freezes a side's momentum when `self.true_single_side` ([poet_lie_orth.py:106](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L106)); for the integrated path (`true_single_side=False`, `alternating=True`) **both** momenta advance. This is the load-bearing difference from the regressed single-side path.
- `megatron_args` only forbids `single_step_x_alternating` + `lie_alternating` ([megatron_args.py:321-325](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L321-L325)); `single_step_x` + `lie_alternating` is allowed.
- `_fold_active_side` is already verified bit-identical to the both-sides fold for the `"in"` side at fp64 ([test_alternating_poetx.py:163-199](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_alternating_poetx.py#L163-L199)); the `"out"` side is currently untested — Task 3 closes that gap.
- `alt_state` is seeded once per training step **before forward** by the merge wrapper ([poet_merge_step.py:99-100](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L99-L100)) and read by `active_side(alternate_every)` ([alt_state.py:26-28](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/alt_state.py#L26-L28)).

---

## Task 1: Phase 1 — champion config, doc, launcher, arg-acceptance test (no `src/` changes)

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth_alt.yaml`
- Create: `docs/experiments/poet_lie_orth_alt.md`
- Create: `scripts/train_poet_lie_orth_alt.sh`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_megatron_args.py`)

```python
def test_single_step_x_with_lie_alternating_emits_both_flags():
    # Integrated path: single_step_x + lie_alternating is ALLOWED (only
    # single_step_x_alternating is mutually exclusive with lie_alternating).
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "parameterization": "cayley",
                "q_optimizer": "lie_ortho",
                "single_step_x": True,
                "lie_alternating": True,
                "single_step_x_alternating": False,
            }
        )
    )
    assert "--poet-single-step-x" in args
    assert "--poet-lie-alternating" in args
    assert "--poet-single-step-x-alternating" not in args


def test_poet_lie_orth_alt_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_orth_alt.yaml")
    assert cfg.experiment.name == "poet_lie_orth_alt"
    assert cfg.optim.poet.q_optimizer == "lie_ortho"
    assert cfg.optim.poet.single_step_x is True
    assert cfg.optim.poet.lie_alternating is True
    assert cfg.optim.poet.single_step_x_alternating is False
    assert cfg.optim.poet.head_aligned_attn is False
    assert cfg.optim.poet.reinit_period == -1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/unit/test_megatron_args.py::test_poet_lie_orth_alt_experiment_yaml -v`
Expected: FAIL — `FileNotFoundError` / config does not exist yet.

(The `*_emits_both_flags` test should already PASS — `_optimizer_args` accepts this combo today. That's intentional: it locks the "allowed" contract so a future validation change can't silently break Phase 1. If it FAILS, a hidden validation blocks the combo — stop and reconcile with [megatron_args.py:287-325](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L287-L325) before continuing.)

- [ ] **Step 3: Create the config** `configs/experiments/optim/poet_lie_orth_alt.yaml`

```yaml
# @package _global_
# poet_lie_orth_alt: the POET champion behavior on the POETX forward-frame layer.
# Plain POETXLinear (single_step_x) + lie_alternating: the optimizer writes ONE
# rotation side per step while BOTH Lie momenta stay fresh (POETXSingleStepFunction
# feeds both grads). Built on the champion lie_ortho recipe (head-OFF, lr 3e-3, c=8,
# distributed). single_step_x_alternating is OFF — this is the integrated both-momenta
# path, NOT the regressed true-single-side path.
# See docs/superpowers/specs/2026-06-08-alternating-poetx-integrated-design.md
#     docs/superpowers/plans/2026-06-08-alternating-poetx-integrated.md
experiment:
  name: poet_lie_orth_alt
  family: optim
  description: |
    Integrated alternating POETX (both-momenta): plain POETXLinear forward-frame
    layer with lie_alternating. The optimizer writes one rotation side per step but
    advances BOTH first-moment momenta every step (Gauss-Seidel-decoupled, 2-step
    accumulated direction). Forward/backward stay both-sides so both momenta stay
    fed — the load-bearing ingredient the regressed true-single-side path
    (poet_lie_orth_alt_x) dropped. Phase-2 adds an active-only merge fold (skip the
    frozen identity side's Cayley) that is bit-identical to the both-sides fold.
    Target: reproduce the lie_ortho+lie_alternating champion (val/loss ≈3.5332) at
    POETX forward speed.
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
    - training_log_eta
    - wandb_metric_normalize
  required_capabilities: []

optim:
  type: poet
  lr: 3.0e-3
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  poet:
    block_count: 1
    cache_mode: none
    init_type: normalized
    mup_alpha: 1.0
    merge_period: 1
    reinit_period: -1
    scale: 0.5
    use_poet_adam: false
    parameterization: cayley
    q_optimizer: lie_ortho
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_ortho_c: 8
    lie_ortho_method: muon
    lie_ortho_ns_steps: 5
    lie_ortho_use_second_moment: false
    lie_ortho_distributed: true
    head_aligned_attn: false
    single_step_fast: true               # forward-frame POETX path
    single_step_x: true                  # forward-frame POETX path
    single_step_x_alternating: false     # integrated both-momenta path (NOT true-single-side)
    lie_alternating: true                # write one side/step, both momenta fresh
    lie_alternate_every: 1
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 4: Create the doc** `docs/experiments/poet_lie_orth_alt.md`

```markdown
# poet_lie_orth_alt

Integrated alternating POETX (both-momenta) on the champion `lie_ortho` recipe
(head-OFF, lr 3e-3, c=8, distributed). Plain `POETXLinear` (`single_step_x`) +
`lie_alternating`: the optimizer writes one rotation side per step while **both**
first-moment momenta stay fresh (`POETXSingleStepFunction` returns both grads). This
is the integrated path, **not** the regressed true-single-side `poet_lie_orth_alt_x`
(which froze the inactive momentum and regressed to 4.22).

- **Design:** `docs/superpowers/specs/2026-06-08-alternating-poetx-integrated-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-08-alternating-poetx-integrated.md`
- **Target:** the `lie_ortho` + `lie_alternating` champion (`1ynrrimu`, val/loss ≈3.5332).
- **Phase 1:** plain POETX, merge folds both sides (frozen side `oft_R=0` → identity →
  no-op) — reproduces the champion at POETX forward speed, zero new code.
- **Phase 2:** active-only merge fold (skip the frozen side's Cayley) — bit-identical
  fold, expected `perf/step_time_s` drop at merge time.
```

- [ ] **Step 5: Create the launcher** `scripts/train_poet_lie_orth_alt.sh`
(Mirror of `scripts/train_poet_lie_orth_alt_x.sh`; only the header comment and the `experiment=` line differ.)

```bash
#!/usr/bin/env bash
set -euo pipefail

# poet_lie_orth_alt: integrated alternating POETX (both-momenta) on the champion
# lie_ortho recipe. Same harness as train_poet_lie_orth.sh, experiment swapped.

case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3"; DEFAULT_SCALE="60m" ;;
  deepseek_v3) FAMILY="deepseek_v3"; DEFAULT_SCALE="deepseek_v3_proxy_small" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

USER_SET_SCALE="no"; USER_SET_SEQ="no"; USER_SET_SCHED="no"; USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    base.model.seq_length=*) USER_SET_SEQ="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=(); [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]] && SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
REGIME_ARGS=(); [[ "${USER_SET_REGIME}" == "no" ]] && REGIME_ARGS=("training_regime=ablation_40x")
SEQ_ARGS=(); [[ "${USER_SET_SEQ}" == "no" ]] && SEQ_ARGS=("base.model.seq_length=256")
SCHED_ARGS=(); [[ "${USER_SET_SCHED}" == "no" ]] && SCHED_ARGS=("scheduler=cosine_poet")

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/poet_lie_orth_alt" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "optim.weight_decay=0.1" \
  "wandb.project=slm-zeju-dev" \
  "$@"
```

- [ ] **Step 6: Make the launcher executable**

Run: `chmod +x scripts/train_poet_lie_orth_alt.sh`

- [ ] **Step 7: Run the tests to verify they pass**

Run: `$PY -m pytest tests/unit/test_megatron_args.py::test_single_step_x_with_lie_alternating_emits_both_flags tests/unit/test_megatron_args.py::test_poet_lie_orth_alt_experiment_yaml -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_alt.yaml docs/experiments/poet_lie_orth_alt.md scripts/train_poet_lie_orth_alt.sh tests/unit/test_megatron_args.py
git commit -m "feat(poet): add integrated alternating POETX champion config (phase 1)"
```

- [ ] **Step 9: Hand the GPU acceptance run to the user (do NOT launch it)**

Print this and stop:

```
GPU acceptance (Phase 1) — run yourself on an H100/A100 node:
  codexlog poet_lie_orth_alt scripts/train_poet_lie_orth_alt.sh
Acceptance: val/loss ≈ 3.5332 at the POETX forward speed (vs the single_step_native
champion 1ynrrimu). This de-risks the whole approach before any layer surgery.
```

---

## Task 2: `POETXLinear` gains `alternating`/`alternate_every` and hosts `_fold_active_side`

The integrated layer needs the `alternating` flag (read by the merge driver in Task 6) and the active-only fold helper on the canonical class. `AlternatingPOETXLinear` (research path) becomes a thin subclass that sets `alternating=True` and keeps its single-side forward.

**Files:**
- Modify: `third_party/poet_torch/poetx_layer.py`
- Test: `tests/unit/test_poetx_layer.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_poetx_layer.py`)

```python
def test_poetx_alternating_flag_defaults_false_and_stores_cadence():
    from poet_torch import POETXLinear

    plain = POETXLinear(in_features=8, out_features=8, block_count=1)
    assert plain.alternating is False
    assert plain.alternate_every == 1

    alt = POETXLinear(in_features=8, out_features=8, block_count=1,
                      alternating=True, alternate_every=3)
    assert alt.alternating is True
    assert alt.alternate_every == 3


def test_alternating_subclass_sets_flag_and_inherits_fold():
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    layer = AlternatingPOETXLinear(in_features=8, out_features=16,
                                   block_count=1, alternate_every=2)
    assert isinstance(layer, POETXLinear)
    assert layer.alternating is True            # routes via the flag in the merge driver
    assert layer.alternate_every == 2
    # _fold_active_side now lives on POETXLinear; the subclass inherits it.
    assert layer._fold_active_side.__qualname__.startswith("POETXLinear.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/unit/test_poetx_layer.py::test_poetx_alternating_flag_defaults_false_and_stores_cadence -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'alternating'`.

- [ ] **Step 3: Add `alternating`/`alternate_every` to `POETXLinear.__init__`**

In `third_party/poet_torch/poetx_layer.py`, change the signature:

```python
    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, parameterization="cayley",
                 alternating=False, alternate_every=1):
```

and immediately after the line `self.single_step_fast = False  # POETX ignores it (forward is always the X op)` add:

```python
        # Alternating both-momenta merge: when set, the merge driver folds ONLY the
        # active side (the frozen side's oft_R is 0 -> identity -> skip its Cayley).
        # Forward/backward stay both-sides (POETXSingleStepFunction feeds BOTH grads),
        # so both momenta stay fed -- the load-bearing ingredient. alternate_every
        # matches the optimizer + alt_state cadence.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
```

- [ ] **Step 4: Move `_fold_active_side` onto `POETXLinear`**

Cut the entire `_fold_active_side` method (currently `third_party/poet_torch/poetx_layer.py:172-201`, on `AlternatingPOETXLinear`) and paste it as a `POETXLinear` method, directly **after** `merge_then_reinitialize`:

```python
    @torch.no_grad()
    def _fold_active_side(self, active, reinit_perm: bool = False, cayley_fn=None) -> None:
        """Fold ONLY the active side into W (skip the frozen side's Cayley build).

        The frozen side's oft_R is exactly 0 => R = I, so its fold is a no-op; we
        build identity blocks for it (no Cayley) and reuse the verified round-trip
        fold. Bit-identical to the both-sides fold whenever the frozen side is
        identity, but pays one Cayley + one block-fold instead of two.
        """
        import torch as _torch
        from .poet_layer import pytorch_skew_symmetric

        if cayley_fn is None:

            def cayley_fn(Q):
                return _torch.ops.poet.cayley(Q)[0]

        if active == "in":
            R_in = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_in, self.block_size_in, self.rows_in, self.cols_in)
            )
            R_out = _torch.eye(self.block_size_out, dtype=self.weight.dtype, device=self.weight.device)
            R_out = R_out.unsqueeze(0).expand(self.r_out, -1, -1).contiguous()  # bmm needs real strides
        else:  # "out"
            R_out = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_out, self.block_size_out, self.rows_out, self.cols_out)
            )
            R_in = _torch.eye(self.block_size_in, dtype=self.weight.dtype, device=self.weight.device)
            R_in = R_in.unsqueeze(0).expand(self.r_in, -1, -1).contiguous()  # bmm needs real strides
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
```

- [ ] **Step 5: Slim `AlternatingPOETXLinear` to a thin subclass**

Replace the whole `AlternatingPOETXLinear` class body so it (a) sets `alternating=True` via super, (b) keeps the single-side forward, and (c) no longer defines `_fold_active_side` or its own `alternate_every` (both now come from the parent):

```python
class AlternatingPOETXLinear(POETXLinear):
    """POETX layer that trains ONE rotation side per step (true single-side).

    The active side comes from the shared `alt_state` iteration (seeded once per
    training step), so layer forward, optimizer, and merge all agree. Forward is
    the unchanged bare GEMM; the backward (AlternatingPOETXSingleStepFunction)
    computes only the active side's rotation-gradient and zeros the frozen side.
    `alternating=True` routes the merge driver to the active-only fold (inherited
    from POETXLinear). This is the gated research path — it freezes the inactive
    side's MOMENTUM (true_single_side optimizer), which regressed quality; the
    integrated both-momenta path uses a plain POETXLinear(alternating=True) instead.
    """

    def __init__(self, *args, alternate_every: int = 1, **kwargs):
        super().__init__(*args, alternating=True, alternate_every=alternate_every, **kwargs)

    def forward(self, x):
        from .alt_state import active_side

        active = active_side(self.alternate_every)
        return AlternatingPOETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, active,
        )
```

- [ ] **Step 6: Run the new tests + the full POETX-layer + existing alternating suites**

Run: `$PY -m pytest tests/unit/test_poetx_layer.py tests/unit/test_alternating_poetx.py -v`
Expected: PASS — including the pre-existing `test_layer_is_poetx_subclass`, `test_active_only_fold_matches_both_sides_when_frozen_is_identity`, and `test_layer_forward_is_bare_gemm_and_backward_is_single_side` (the subclass forward is unchanged; `alternate_every`/`_fold_active_side` are now inherited).

- [ ] **Step 7: Commit**

```bash
git add third_party/poet_torch/poetx_layer.py tests/unit/test_poetx_layer.py
git commit -m "feat(poetx): host alternating flag + _fold_active_side on POETXLinear"
```

---

## Task 3: Close the `_fold_active_side` `"out"`-side fp64 parity gap

The `"in"` side is verified bit-identical; the `"out"` side is untested (the Task-7 gap). Prove both sides of `POETXLinear._fold_active_side` equal the both-sides fold when the frozen side is identity.

**Files:**
- Test: `tests/unit/test_poetx_layer.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_poetx_layer.py`)

```python
import pytest


@pytest.mark.parametrize("active", ["in", "out"])
def test_fold_active_side_matches_both_sides_when_frozen_is_identity(active):
    """POETXLinear._fold_active_side(active) == the both-sides fold when the frozen
    side's oft_R is 0 (identity). Closes the Task-7 'out'-side gap. fp64."""
    import torch
    from poet_torch import POETXLinear
    from poet_torch.poet_layer import cayley_batch, pytorch_skew_symmetric

    torch.set_default_dtype(torch.float64)
    try:
        def _make():
            # Seed INSIDE so ref and act are bit-identical clones (same perms,
            # weight, and the active side's oft_R). The frozen side stays 0.
            torch.manual_seed(5)
            layer = POETXLinear(in_features=12, out_features=8, block_count=1,
                                bias=False, alternating=True)
            with torch.no_grad():
                layer.weight.normal_()
                if active == "in":
                    layer.oft_R_in.normal_(std=1e-2)   # oft_R_out left at 0 (identity)
                else:
                    layer.oft_R_out.normal_(std=1e-2)  # oft_R_in left at 0 (identity)
            return layer

        def _cayley(layer):
            qi = pytorch_skew_symmetric(
                layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in
            )
            qo = pytorch_skew_symmetric(
                layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out
            )
            return cayley_batch(qo), cayley_batch(qi)  # (R_out, R_in)

        ref, act = _make(), _make()
        R_out, R_in = _cayley(ref)
        ref._fold_with_R(R_out, R_in, reinit_perm=False)        # full both-sides fold
        act._fold_active_side(active, reinit_perm=False, cayley_fn=cayley_batch)
        assert torch.allclose(act.weight, ref.weight, atol=1e-9), \
            (act.weight - ref.weight).abs().max()
        # the active side's oft_R is zeroed by the fold; the frozen side was already 0
        assert torch.count_nonzero(act.oft_R_in) == 0
        assert torch.count_nonzero(act.oft_R_out) == 0
    finally:
        torch.set_default_dtype(torch.float32)
```

- [ ] **Step 2: Run to verify it passes** (the implementation already exists from Task 2; this is the missing coverage)

Run: `$PY -m pytest "tests/unit/test_poetx_layer.py::test_fold_active_side_matches_both_sides_when_frozen_is_identity" -v`
Expected: PASS for both `[in]` and `[out]` params.

(If `[out]` FAILS, the `"out"` branch of `_fold_active_side` is wrong — this is exactly the gap the task exists to catch. Debug `_fold_active_side`'s `"out"` branch with superpowers:systematic-debugging before proceeding.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_poetx_layer.py
git commit -m "test(poetx): close _fold_active_side out-side fp64 parity gap"
```

---

## Task 4: Optimizer `_active_side` reads `alt_state` in `alternating` mode

The optimizer's *write* side and the merge's *fold* side must come from the same source. Switch the `alternating` branch from the internal `_alt_step` counter to the shared `alt_state` iteration.

**Files:**
- Modify: `src/optim/poet_lie_orth.py:91-98` (`_active_side`)
- Test: `tests/unit/test_poet_lie_orth.py:267-292` (update existing)

- [ ] **Step 1: Update the existing test to seed `alt_state`** — replace `test_batched_step_alternating_writes_only_active_side` in `tests/unit/test_poet_lie_orth.py` with:

```python
def test_batched_step_alternating_writes_only_active_side():
    # Alternating now reads the SHARED alt_state iteration (not the internal
    # _alt_step counter). The iterations are chosen so they do NOT coincide with the
    # internal counter (which would start at 0): iteration 1 -> active 'in',
    # iteration 2 -> active 'out'. Momentum accrues on BOTH sides; only the active
    # side's oft_R is written. This sequence FAILS against the old _alt_step source
    # (step 1 would pick 'out') and PASSES against the alt_state source.
    from poet_torch import alt_state

    torch.manual_seed(2)
    b = 8
    ne = b * (b - 1) // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.randn(1, ne)
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_out.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
        alternating=True,
    )
    alt_state.set_iteration(1)  # active 'in' (old _alt_step=0 would pick 'out')
    opt.step()
    assert p_in.data.abs().sum() > 0 and torch.allclose(p_out.data, torch.zeros_like(p_out))
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.randn(1, ne)
    p_in.data.zero_()  # simulate the per-step fold of the just-written side
    alt_state.set_iteration(2)  # active 'out' (old _alt_step=1 would pick 'in')
    opt.step()
    assert p_out.data.abs().sum() > 0 and torch.allclose(p_in.data, torch.zeros_like(p_in))
    alt_state.set_iteration(0)  # restore the module global for later tests
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/unit/test_poet_lie_orth.py::test_batched_step_alternating_writes_only_active_side -v`
Expected: FAIL on the **first** assertion — with the old code the `alternating` source is the internal `_alt_step` counter (0 on the first step → `(0//1)%2==0` → `out`), so `p_out` is written and `p_in` stays zero, contradicting the `set_iteration(1)` → `in` expectation. The `set_iteration` calls are no-ops for the old code, so this is the step that proves the source actually changed.

- [ ] **Step 3: Change `_active_side`** in `src/optim/poet_lie_orth.py` — replace the method body (lines 91-98):

```python
    def _active_side(self):
        # The dedicated true-single-side path AND the integrated both-momenta
        # alternating path both read the SAME shared signal (alt_state, seeded
        # once per training step by the poet_merge_step wrapper) so the optimizer's
        # WRITE side equals the merge's FOLD side within a step. Quality-neutral
        # for the both-sides-merge case; REQUIRED for active-only-merge correctness.
        if self.true_single_side or self.alternating:
            from poet_torch.alt_state import active_side

            return active_side(self.alternate_every)
        return None
```

(Leave the `self._alt_step` field and its end-of-`step()` increment as-is — now vestigial for the `alternating` path but harmless; removing them is out of scope.)

- [ ] **Step 4: Run the updated test + the full optimizer suite**

Run: `$PY -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: PASS — the updated alternating test plus the existing `test_true_single_side_*` tests (those already seed `alt_state`).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_lie_orth.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): alternating optimizer reads alt_state for active side"
```

---

## Task 5: Walk builds `POETXLinear(alternating=True)` for `single_step_x` + `lie_alternating`

`replace_linears_with_poet` learns a `lie_alternating` flag and forwards `alternating=lie_alternating` into the `POETXLinear` it builds.

**Files:**
- Modify: `src/optim/poet_layers.py:179-199` (signature) and `:314-343` (construction)
- Test: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_poet_layers.py`)

```python
def test_single_step_x_with_lie_alternating_builds_alternating_poetx():
    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        lie_alternating=True,
        alternate_every=2,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    # Integrated path: a PLAIN POETXLinear with the alternating flag set -- NOT the
    # true-single-side AlternatingPOETXLinear subclass (both momenta stay fed).
    assert isinstance(pl, POETXLinear)
    assert not isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternating is True
    assert pl.alternate_every == 2


def test_single_step_x_without_lie_alternating_builds_plain_poetx():
    import torch.nn as nn
    from poet_torch import POETXLinear

    from src.optim.poet_layers import replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m, block_count=1, init_type="none",
        extra_linear_types=(nn.Linear,), single_step_x=True,
    )
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear)
    assert pl.alternating is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/unit/test_poet_layers.py::test_single_step_x_with_lie_alternating_builds_alternating_poetx -v`
Expected: FAIL — `TypeError: replace_linears_with_poet() got an unexpected keyword argument 'lie_alternating'`.

- [ ] **Step 3: Add the `lie_alternating` parameter** to `replace_linears_with_poet` in `src/optim/poet_layers.py` — add it right after `single_step_x_alternating: bool = False,` in the signature:

```python
    single_step_x_alternating: bool = False,
    lie_alternating: bool = False,
    alternate_every: int = 1,
```

- [ ] **Step 4: Restructure the `cache_mode == "none"` construction** in `src/optim/poet_layers.py`. Replace the existing `if single_step_x and single_step_x_alternating: ... else: ...` block (currently lines 315-343) with:

```python
                if cache_mode == "none":
                    if single_step_x and single_step_x_alternating:
                        from poet_torch import AlternatingPOETXLinear as _PoetCls

                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    elif single_step_x:
                        # Integrated path: a plain POETXLinear that carries the
                        # alternating flag (both-momenta forward/backward; the merge
                        # driver folds only the active side). lie_alternating=False
                        # builds the ordinary both-sides POETXLinear.
                        from poet_torch import POETXLinear as _PoetCls

                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternating=lie_alternating,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    else:
                        if single_step_native:
                            from poet_torch import SingleStepPOETLinear as _PoetCls
                        else:
                            _PoetCls = POETLinear  # noqa: N806
                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            **block_kwargs,
                        )
                else:
```

(The `else:` on the last line is the existing `CachedPOETLinear` branch — leave it and everything below unchanged. `pl.bake_perms_into_weight()` at line ~365 still runs for any `single_step_x` layer, including the new alternating one.)

- [ ] **Step 5: Run the new tests + the full walk-selection suite**

Run: `$PY -m pytest tests/unit/test_poet_layers.py -v`
Expected: PASS — including the pre-existing `test_single_step_x_uses_poetx_class`, `test_single_step_x_alternating_uses_alternating_poetx_class`, `test_single_step_native_uses_new_class`.

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): walk builds POETXLinear(alternating=True) for single_step_x+lie_alternating"
```

---

## Task 6: Thread `lie_alternating` through the apply patch

The model-build patch must read `args.poet_lie_alternating` and pass it to the walk, or the layer is built without the flag at runtime.

**Files:**
- Modify: `src/patches/poet_apply_to_model.py:61-96` (`_apply_poet_to_chunk`)
- Test: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_poet_layers.py`) — drive `_apply_poet_to_chunk` with a stub args object, no Megatron needed:

```python
def test_apply_patch_threads_lie_alternating_into_walk():
    import types

    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    import src.patches.poet_apply_to_model as ap

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    # Minimal args carrying just the POET knobs _apply_poet_to_chunk reads.
    args = types.SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        poet_init_type="none",
        poet_mup_alpha=1.0,
        poet_cache_mode="none",
        poet_parameterization="cayley",
        poet_freeze_output_rotation=False,
        poet_head_aligned_attn=False,
        poet_no_head_resid_perm=False,
        poet_single_step_fast=True,
        poet_single_step_native=False,
        poet_single_step_x=True,
        poet_single_step_x_alternating=False,
        poet_lie_alternating=True,
        poet_lie_alternate_every=3,
        kv_channels=None,
        hidden_size=8,
        num_attention_heads=1,
    )
    # _apply_poet_to_chunk discovers Megatron linear types lazily; on a CPU node that
    # returns () and the walk falls back to extra_linear_types. We can't pass
    # extra_linear_types through the patch, so monkeypatch the walk to assert the
    # flag is forwarded, then build for real.
    seen = {}
    orig = ap.replace_linears_with_poet

    def _spy(model, **kw):
        seen.update(kw)
        return orig(model, extra_linear_types=(nn.Linear,), **kw)

    ap.replace_linears_with_poet = _spy
    try:
        m = M()
        ap._apply_poet_to_chunk(m, args)
    finally:
        ap.replace_linears_with_poet = orig

    assert seen["lie_alternating"] is True
    assert seen["alternate_every"] == 3
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear) and not isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternating is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/unit/test_poet_layers.py::test_apply_patch_threads_lie_alternating_into_walk -v`
Expected: FAIL — `KeyError: 'lie_alternating'` (the patch does not yet pass it).

- [ ] **Step 3: Thread the flag** in `src/patches/poet_apply_to_model.py`. In `_apply_poet_to_chunk`, after the line `single_step_x_alternating = getattr(args, "poet_single_step_x_alternating", False)` add:

```python
        lie_alternating = getattr(args, "poet_lie_alternating", False)
```

and in the `return replace_linears_with_poet(...)` call, add the argument next to `single_step_x_alternating=single_step_x_alternating,`:

```python
            single_step_x_alternating=single_step_x_alternating,
            lie_alternating=lie_alternating,
            alternate_every=alternate_every,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/unit/test_poet_layers.py::test_apply_patch_threads_lie_alternating_into_walk -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_apply_to_model.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): thread lie_alternating through the apply patch"
```

---

## Task 7: Merge driver routes the active-only fold by the `alternating` flag

The final integration: `_merge_layers` folds active-only for any layer with `alternating=True` (the integrated `POETXLinear(alternating=True)` AND the research `AlternatingPOETXLinear`), instead of by `isinstance`. This is the step that turns on the POETX-native speedup and makes the integrated layer fold the same side the optimizer wrote.

**Files:**
- Modify: `src/patches/poet_merge_step.py:386-413` (`_merge_layers`)
- Test: `tests/unit/test_alternating_poetx.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_alternating_poetx.py`)

```python
def test_merge_layers_routes_integrated_poetx_to_active_only_fold(monkeypatch):
    from poet_torch import POETXLinear, alt_state

    import src.patches.poet_merge_step as ms

    alt_state.set_iteration(1)  # active "in" at alternate_every=1
    layer = POETXLinear(in_features=8, out_features=8, block_count=1, bias=False,
                        alternating=True)
    calls = []
    monkeypatch.setattr(
        POETXLinear,
        "_fold_active_side",
        lambda self, side, reinit_perm=False: calls.append(side),
    )
    ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
    assert calls == ["in"]
    alt_state.set_iteration(0)


def test_merge_layers_keeps_plain_poetx_on_both_sides_fold(monkeypatch):
    # alternating=False must NOT route to _fold_active_side.
    from poet_torch import POETXLinear, alt_state

    import src.patches.poet_merge_step as ms

    alt_state.set_iteration(1)
    layer = POETXLinear(in_features=8, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        layer.weight.normal_()
    active_calls = []
    monkeypatch.setattr(
        POETXLinear,
        "_fold_active_side",
        lambda self, side, reinit_perm=False: active_calls.append(side),
    )
    ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
    assert active_calls == []  # plain POETX went through the batched both-sides fold
    alt_state.set_iteration(0)


def test_integrated_alternating_write_side_matches_fold_side():
    """End-to-end consistency: the side the optimizer WRITES each step (driven by
    alt_state) equals the side the merge driver FOLDS (same alt_state)."""
    from poet_torch import POETXLinear, alt_state
    from poet_torch.alt_state import active_side

    import src.patches.poet_merge_step as ms
    from src.optim.poet_lie_orth import LieOrthMomentum

    torch.manual_seed(0)
    layer = POETXLinear(in_features=8, out_features=8, block_count=1, bias=False,
                        alternating=True, alternate_every=1)
    with torch.no_grad():
        layer.weight.normal_()
    opt = LieOrthMomentum(
        [
            dict(params=[layer.oft_R_in], use_skew=True, side="in", lr=0.1),
            dict(params=[layer.oft_R_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        alternating=True,  # integrated both-momenta path (true_single_side stays False)
    )
    folded = []
    _real_fold = POETXLinear._fold_active_side

    def _spy_fold(self, side, reinit_perm=False):
        folded.append(side)
        return _real_fold(self, side, reinit_perm=reinit_perm)

    POETXLinear._fold_active_side = _spy_fold
    try:
        for it in range(1, 5):
            alt_state.set_iteration(it)
            layer.oft_R_in.grad = torch.randn_like(layer.oft_R_in)
            layer.oft_R_out.grad = torch.randn_like(layer.oft_R_out)
            opt.step()  # writes active_side(it) only; BOTH momenta advanced
            active = active_side(it)
            wrote_in = layer.oft_R_in.abs().sum().item() > 0
            wrote_out = layer.oft_R_out.abs().sum().item() > 0
            assert (wrote_in, wrote_out) == ((active == "in"), (active == "out")), (it, active)
            ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
            assert folded[-1] == active, (it, active, folded[-1])
            # the fold zeroed both sides -> next step starts clean
            assert layer.oft_R_in.abs().sum().item() == 0
            assert layer.oft_R_out.abs().sum().item() == 0
    finally:
        POETXLinear._fold_active_side = _real_fold
        alt_state.set_iteration(0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/unit/test_alternating_poetx.py::test_merge_layers_routes_integrated_poetx_to_active_only_fold -v`
Expected: FAIL — the current `_merge_layers` routes by `isinstance(pl, AlternatingPOETXLinear)`, so a plain `POETXLinear(alternating=True)` is treated as `rest` and never calls `_fold_active_side` (`calls == []`).

- [ ] **Step 3: Switch `_merge_layers` to flag-based routing** — replace the whole function in `src/patches/poet_merge_step.py` (lines 386-413):

```python
def _merge_layers(pls, reinit_perm: bool, disable_batch: bool) -> None:
    """Fold every layer. Layers with ``alternating=True`` (the integrated
    POETXLinear both-momenta path AND the research AlternatingPOETXLinear subclass)
    fold ONLY the active side -- the frozen side's oft_R is 0 (identity), so its
    Cayley + fold are skipped. The active side comes from each layer's OWN
    alternate_every via alt_state (no megatron get_args, so this stays
    importable/callable on CPU). The rest use the batched both-sides fold."""
    from poet_torch.alt_state import active_side

    alt_pls = [pl for pl in pls if getattr(pl, "alternating", False)]
    rest = [pl for pl in pls if not getattr(pl, "alternating", False)]

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
```

(This drops the now-unused `from poet_torch import AlternatingPOETXLinear` import that was inside the old function — it is no longer referenced.)

- [ ] **Step 4: Run the new tests + the full alternating + merge-step suites**

Run: `$PY -m pytest tests/unit/test_alternating_poetx.py tests/unit/test_poet_merge_step.py -v`
Expected: PASS — including the pre-existing `test_merge_layers_routes_alternating_to_active_only_fold` (the research `AlternatingPOETXLinear` now has `alternating=True`, so the flag-based router still catches it).

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_alternating_poetx.py
git commit -m "feat(poet): route active-only merge fold by the alternating flag"
```

---

## Task 8: Full CPU suite green + GPU acceptance handoff

**Files:** none (verification only)

- [ ] **Step 1: Run the whole POET-related CPU suite**

Run:
```bash
$PY -m pytest \
  tests/unit/test_poetx_layer.py \
  tests/unit/test_alternating_poetx.py \
  tests/unit/test_poet_layers.py \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_merge_step.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_alt_state.py \
  -q
```
Expected: all pass (0 failures). Per the test-env memo, 2 unrelated `launchers.submit` failures may exist elsewhere in the repo — they are pre-existing and NOT in the files above; if they appear, confirm they predate this branch via `git stash && rerun` before dismissing.

- [ ] **Step 2: Static check the touched Python files**

Run:
```bash
$PY -m py_compile \
  third_party/poet_torch/poetx_layer.py \
  src/optim/poet_lie_orth.py \
  src/optim/poet_layers.py \
  src/patches/poet_apply_to_model.py \
  src/patches/poet_merge_step.py
ruff check third_party/poet_torch/poetx_layer.py src/optim/poet_lie_orth.py src/optim/poet_layers.py src/patches/poet_apply_to_model.py src/patches/poet_merge_step.py
```
Expected: no errors. (If `ruff` is not on PATH, use `$PY -m ruff check ...`.)

- [ ] **Step 3: Confirm the research path is untouched in behavior**

Run: `$PY -m pytest tests/unit/test_megatron_args.py -k "single_step_x_alternating" -v`
Expected: PASS — `single_step_x_alternating` still emits its flag and stays mutually exclusive with `lie_alternating` ([megatron_args.py:321-325](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L321-L325)).

- [ ] **Step 4: Update the CHANGELOG** (per the repo's logging convention) — append a dated entry to `NeckariumAI/zqiu/CHANGELOG.md` summarizing: integrated alternating POETX (both-momenta) — Phase-1 champion config `poet_lie_orth_alt`, Phase-2 active-only merge fold on `POETXLinear(alternating=True)`.

- [ ] **Step 5: Commit the CHANGELOG**

```bash
git add NeckariumAI/zqiu/CHANGELOG.md
git commit -m "docs: changelog for integrated alternating POETX"
```

- [ ] **Step 6: Hand the GPU acceptance to the user (do NOT launch)**

Print and stop:

```
GPU acceptance — run yourself on an H100/A100 node:

  # Phase 1 already validated the recipe; Phase 2 must keep the same loss + drop merge time.
  codexlog poet_lie_orth_alt scripts/train_poet_lie_orth_alt.sh

Acceptance:
  * val/loss stays ≈ 3.5332 (active-only fold is bit-identical to both-sides).
  * perf/step_time_s shows a merge-time drop vs the Phase-1 both-sides fold
    (one Cayley + one block-fold per layer instead of two).
Optional A/B for the merge-time delta: flip alternating off via
  scripts/train_poet_lie_orth_alt.sh optim.poet.lie_alternating=false
and compare perf/step_time_s at matched config.
```

---

## Self-Review

**Spec coverage**

| Spec item | Task |
|---|---|
| Phase 1 — champion config (`single_step_x` + `lie_alternating`, head-off, lr 3e-3, c=8, distributed, `single_step_x_alternating=false`) | Task 1 |
| Phase 1 — doc + launcher | Task 1 |
| Phase 1 — GPU acceptance ≈3.5332 at POETX speed | Task 1 Step 9 |
| Phase 2.1 — `POETXLinear` gains `alternating`/`alternate_every`; hosts `_fold_active_side` | Task 2 |
| Phase 2.2 — optimizer `alternating` reads `alt_state` (`_active_side`) | Task 4 |
| Phase 2.3 — walk builds `POETXLinear(alternating=True)` | Task 5 |
| Phase 2.3 — `poet_apply_to_model` threads `lie_alternating` | Task 6 |
| Phase 2.4 — `_merge_layers` routes by the `alternating` flag | Task 7 |
| Active-side data flow (write side == fold side) | Task 7 Step 1 (`test_integrated_alternating_write_side_matches_fold_side`) |
| Research path kept, gated off (`AlternatingPOETXLinear`, `true_single_side`, the flag) | Task 2 (thin subclass), Task 8 Step 3 (validation intact) |
| Test — fold parity both `"in"`/`"out"`, fp64 (closes Task-7 gap) | Task 3 |
| Test — optimizer alt reads `alt_state` | Task 4 |
| Test — write==fold consistency | Task 7 |
| Test — walk-selection (`single_step_x`+`lie_alternating` → integrated; `single_step_x_alternating` → research) | Task 5 (integrated) + existing `test_single_step_x_alternating_uses_alternating_poetx_class` (research) |
| Full CPU suite green | Task 8 |
| GPU — Phase 2 keeps 3.5332 + merge-time drop | Task 8 Step 6 |
| Out of scope: d³ backward saving; one-sided fold matmul (Task 7 v2) | not implemented, by design |

**Type/name consistency** — `alternating`, `alternate_every`, `_fold_active_side(active, reinit_perm=…, cayley_fn=…)`, `active_side(alternate_every)`, `lie_alternating`, `_active_side()`, `_merge_layers(pls, reinit_perm, disable_batch)` are used identically across every task and match the current code signatures. `POETXLinear.forward` is deliberately **not** branched on `alternating` (both-momenta backward is the whole point); only the *merge driver* and the *optimizer write* branch on the active side.

**Placeholder scan** — no TBD / "add error handling" / "write tests for the above" / "similar to Task N": every code and test step contains complete, runnable content.
