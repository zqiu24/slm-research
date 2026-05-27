# Decoupled LR-Scheduler Config Axis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every LR scheduler (cosine, WSD, constant, linear, inverse-square-root) genuinely run, selectable as a first-class Hydra config axis, and remove the unused step-decay legacy.

**Architecture:** A new pure resolver `src/utils/scheduler.py` translates a `scheduler:` config block into Megatron LR/decay CLI flags; `_training_args` calls it in one line. Scheduler becomes its own Hydra group `configs/scheduler/` (one file per type), decoupled from `training_regime`. WSD is Megatron-native — no patches. Decay-only resume is two-run, flag-only.

**Tech Stack:** Python 3.12, OmegaConf/Hydra config composition, pytest (CPU-only), Megatron-LM (`third_party/Megatron-LM`).

**Spec:** [docs/superpowers/specs/2026-05-27-scheduler-config-axis-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-27-scheduler-config-axis-design.md)

**Reference — Megatron-native flags this plan emits:**
- `--lr-decay-style` ∈ `{constant, linear, cosine, inverse-square-root, WSD}` (note `WSD` is uppercase)
- `--lr-wsd-decay-style` ∈ `{exponential, linear, cosine, minus_sqrt}`, `--lr-wsd-decay-samples N`
- `--lr-warmup-fraction F` **or** `--lr-warmup-samples N` (mutually exclusive)
- `--min-lr V`, `--lr-decay-samples N`
- `--finetune`, `--override-opt-param-scheduler` (decay-only resume)

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 0.1: Commit the pending 300m train-script edits first (they belong to the testing-default theme, not this feature)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add scripts/train_adam.sh scripts/train_muon.sh scripts/train_ngpt.sh
git commit -m "chore(scripts): default llama3 train scripts to 300m scale"
```

- [ ] **Step 0.2: Create the feature branch off main**

```bash
git checkout main
git checkout -b scheduler-config-axis
```

If `git checkout main` complains about the uncommitted `third_party/poet_torch/poet_layer.py` edit (the pre-existing `cayley_batch` change, unrelated to this work), stash it: `git stash push third_party/poet_torch/poet_layer.py` and restore later. Do **not** commit it as part of this feature.

- [ ] **Step 0.3: Confirm the suite is green at baseline**

Run: `pytest tests/unit/test_megatron_args.py -q`
Expected: PASS (the engineer runs tests on the cluster; this plan's tests are all CPU-only).

---

## Task 1: Remove step-decay legacy

Megatron has no native `step` decay style; the repo added it via a patch we no longer want. This task deletes it cleanly. It is self-contained — nothing else depends on it (verified by grep; the `poet.py` "step" hits are the optimizer step *counter*, unrelated).

**Files:**
- Delete: `src/patches/lr_decay_style_step.py`
- Delete: `configs/experiments/optim/adamw_step_decay.yaml`
- Delete: `tests/unit/test_megatron_args_step_decay.py`
- Modify: `launchers/pretrain_gpt_slm.py:63-66`
- Modify: `src/utils/megatron_args.py:160-168`

- [ ] **Step 1.1: Delete the three step-decay files**

```bash
git rm src/patches/lr_decay_style_step.py \
       configs/experiments/optim/adamw_step_decay.yaml \
       tests/unit/test_megatron_args_step_decay.py
```

- [ ] **Step 1.2: Remove the step CLI args from the launcher**

In `launchers/pretrain_gpt_slm.py`, delete these four lines (currently 63-66):

```python
    # Piecewise-constant step decay (see src/patches/lr_decay_style_step.py).
    # Only consumed when --lr-decay-style=step.
    group.add_argument("--lr-decay-step-ratio", nargs="+", type=float, default=None)
    group.add_argument("--lr-decay-step-coeff", nargs="+", type=float, default=None)
```

(Leave the `--ngpt-no-warmup` argument above them untouched.)

- [ ] **Step 1.3: Remove the inline step branch from `_training_args`**

In `src/utils/megatron_args.py`, delete the step branch (currently lines 160-168):

```python
    if lr_decay_style == "step":
        ratio = training.get("lr_decay_step_ratio", None)
        coeff = training.get("lr_decay_step_coeff", None)
        if ratio is None or coeff is None:
            raise ValueError(
                "training.lr_decay_style=step requires training.lr_decay_step_ratio "
                "and training.lr_decay_step_coeff"
            )
        args.append("--lr-decay-step-ratio")
        args.extend(str(float(r)) for r in ratio)
        args.append("--lr-decay-step-coeff")
        args.extend(str(float(c)) for c in coeff)
```

The lines around it (`lr_decay_style = str(training.get("lr_decay_style", "cosine"))` and `_add(args, "--lr-decay-style", lr_decay_style)`) stay for now — they're replaced in Task 7.

- [ ] **Step 1.4: Verify the suite is still green (minus the deleted test)**

Run: `pytest tests/unit/test_megatron_args.py -q`
Expected: PASS. The remaining tests don't reference step decay.

- [ ] **Step 1.5: Commit**

```bash
git add -A
git commit -m "refactor(scheduler): remove unused step-decay legacy"
```

---

## Task 2: Scheduler resolver — happy paths (TDD)

Create the pure resolver and prove each Megatron-native style emits the right flags. No Megatron import — testable on a laptop.

**Files:**
- Create: `src/utils/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 2.1: Write the failing happy-path tests**

Create `tests/unit/test_scheduler.py`:

```python
"""Unit tests for src/utils/scheduler.py (pure config -> Megatron flags)."""

from __future__ import annotations

import pytest

from src.utils.scheduler import scheduler_args


def _to_map(args: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


# total_tokens=6_000_000_000, seq_length=4096 -> 1_464_843 train samples.
_TOTAL = 6_000_000_000
_SEQ = 4096
_PEAK = 1.0e-3


def test_cosine_emits_decay_style_warmup_fraction_and_min_lr():
    m = _to_map(scheduler_args(
        {"type": "cosine", "warmup_fraction": 0.01, "min_lr_ratio": 0.1},
        peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ,
    ))
    assert m["--lr-decay-style"] == "cosine"
    assert m["--lr-warmup-fraction"] == "0.01"
    assert m["--min-lr"] == str(1.0e-3 * 0.1)


def test_constant_has_no_min_lr_and_no_decay_tail():
    m = _to_map(scheduler_args(
        {"type": "constant", "warmup_fraction": 0.01},
        peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ,
    ))
    assert m["--lr-decay-style"] == "constant"
    assert "--min-lr" not in m
    assert "--lr-wsd-decay-samples" not in m


def test_wsd_emits_wsd_style_and_tail_samples():
    m = _to_map(scheduler_args(
        {"type": "wsd", "warmup_fraction": 0.01, "min_lr_ratio": 0.1,
         "wsd_decay_fraction": 0.2, "wsd_decay_style": "cosine"},
        peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ,
    ))
    assert m["--lr-decay-style"] == "WSD"
    assert m["--lr-wsd-decay-style"] == "cosine"
    # round(0.2 * 6e9) // 4096
    assert m["--lr-wsd-decay-samples"] == str(int(round(0.2 * _TOTAL)) // _SEQ)


def test_inverse_square_root_type_is_normalized():
    m = _to_map(scheduler_args(
        {"type": "inverse_square_root", "warmup_fraction": 0.01, "min_lr_ratio": 0.1},
        peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ,
    ))
    assert m["--lr-decay-style"] == "inverse-square-root"


def test_warmup_tokens_converts_to_samples():
    m = _to_map(scheduler_args(
        {"type": "cosine", "warmup_tokens": 60_000_000, "min_lr_ratio": 0.1},
        peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ,
    ))
    assert m["--lr-warmup-samples"] == str(60_000_000 // _SEQ)
    assert "--lr-warmup-fraction" not in m
```

- [ ] **Step 2.2: Run to verify failure**

Run: `pytest tests/unit/test_scheduler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.scheduler'`.

- [ ] **Step 2.3: Implement the resolver**

Create `src/utils/scheduler.py`:

```python
"""Resolve a `scheduler:` config block into Megatron LR/decay CLI flags.

Pure function — no Megatron import, no torch. Unit-testable on CPU.
See docs/superpowers/specs/2026-05-27-scheduler-config-axis-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# config type (normalized, dashes->underscores, lowercased) -> Megatron value
_DECAY_STYLE = {
    "constant": "constant",
    "linear": "linear",
    "cosine": "cosine",
    "inverse_square_root": "inverse-square-root",
    "wsd": "WSD",
}
_VALID_WSD_DECAY_STYLES = ("exponential", "linear", "cosine", "minus_sqrt")

_COMMON_FIELDS = {"type", "warmup_fraction", "warmup_tokens", "min_lr_ratio"}
# Extra fields each type may set, beyond the common ones.
_TYPE_EXTRA_FIELDS = {
    "constant": set(),
    "linear": set(),
    "cosine": set(),
    "inverse_square_root": set(),
    "wsd": {"wsd_decay_fraction", "wsd_decay_style"},
}


def _norm_type(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    if key not in _DECAY_STYLE:
        raise ValueError(
            f"unknown scheduler type {raw!r}; valid: "
            "constant, linear, cosine, inverse-square-root, wsd"
        )
    return key


def _warmup_args(
    sched: Mapping[str, Any], *, seq_length: int, force_no_warmup: bool
) -> list[str]:
    has_frac = sched.get("warmup_fraction", None) is not None
    has_tok = sched.get("warmup_tokens", None) is not None
    if has_frac and has_tok:
        raise ValueError("set only one of warmup_fraction / warmup_tokens, not both")
    if force_no_warmup:
        return ["--lr-warmup-samples", "0"]
    if has_tok:
        wt = int(sched["warmup_tokens"])
        if wt < 0:
            raise ValueError(f"warmup_tokens must be >= 0, got {wt}")
        return ["--lr-warmup-samples", str(wt // seq_length)]
    wf = float(sched.get("warmup_fraction", 0.0))
    if not (0.0 <= wf < 1.0):
        raise ValueError(f"warmup_fraction must be in [0, 1), got {wf}")
    return ["--lr-warmup-fraction", str(wf)]


def scheduler_args(
    sched: Mapping[str, Any],
    *,
    peak_lr: float,
    total_tokens: int,
    seq_length: int,
    force_no_warmup: bool = False,
) -> list[str]:
    """Validate `sched` and return the Megatron LR/decay CLI flags.

    `peak_lr` is cfg.optim.lr (emitted separately as --lr by the caller);
    here it only scales min_lr_ratio. `force_no_warmup` lets the caller
    preserve nGPT's no-LR-warmup behavior regardless of the scheduler file.
    """
    if "type" not in sched:
        raise ValueError("scheduler block missing required field 'type'")
    norm = _norm_type(str(sched["type"]))

    allowed = _COMMON_FIELDS | _TYPE_EXTRA_FIELDS[norm]
    foreign = set(sched.keys()) - allowed
    if foreign:
        raise ValueError(
            f"scheduler type {norm!r} does not accept field(s) {sorted(foreign)}; "
            f"allowed: {sorted(allowed)}"
        )

    args: list[str] = ["--lr-decay-style", _DECAY_STYLE[norm]]

    if norm != "constant":
        min_lr_ratio = sched.get("min_lr_ratio", None)
        if min_lr_ratio is not None:
            args += ["--min-lr", str(float(min_lr_ratio) * float(peak_lr))]

    args += _warmup_args(sched, seq_length=seq_length, force_no_warmup=force_no_warmup)

    if norm == "wsd":
        wdf = sched.get("wsd_decay_fraction", None)
        if wdf is None:
            raise ValueError("scheduler type 'wsd' requires 'wsd_decay_fraction'")
        wdf = float(wdf)
        if not (0.0 < wdf <= 1.0):
            raise ValueError(f"wsd_decay_fraction must be in (0, 1], got {wdf}")
        wsd_decay_samples = int(round(wdf * total_tokens)) // seq_length
        args += ["--lr-wsd-decay-samples", str(wsd_decay_samples)]
        wsd_style = str(sched.get("wsd_decay_style", "cosine"))
        if wsd_style not in _VALID_WSD_DECAY_STYLES:
            raise ValueError(
                f"wsd_decay_style must be one of {list(_VALID_WSD_DECAY_STYLES)}, "
                f"got {wsd_style!r}"
            )
        args += ["--lr-wsd-decay-style", wsd_style]

    return args
```

- [ ] **Step 2.4: Run to verify pass**

Run: `pytest tests/unit/test_scheduler.py -q`
Expected: PASS (5 tests).

- [ ] **Step 2.5: Commit**

```bash
git add src/utils/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): pure resolver from scheduler block to Megatron flags"
```

---

## Task 3: Scheduler resolver — validation (TDD)

Prove the resolver rejects bad config loudly (this is what kills the phantom-config bug class).

**Files:**
- Test: `tests/unit/test_scheduler.py` (extend)

- [ ] **Step 3.1: Add failing validation tests**

Append to `tests/unit/test_scheduler.py`:

```python
def _call(sched, **kw):
    base = dict(peak_lr=_PEAK, total_tokens=_TOTAL, seq_length=_SEQ)
    base.update(kw)
    return scheduler_args(sched, **base)


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown scheduler type"):
        _call({"type": "triangular", "warmup_fraction": 0.01})


def test_missing_type_raises():
    with pytest.raises(ValueError, match="missing required field 'type'"):
        _call({"warmup_fraction": 0.01})


def test_wsd_missing_decay_fraction_raises():
    with pytest.raises(ValueError, match="requires 'wsd_decay_fraction'"):
        _call({"type": "wsd", "warmup_fraction": 0.01, "min_lr_ratio": 0.1})


def test_both_warmup_forms_raises():
    with pytest.raises(ValueError, match="only one of warmup_fraction"):
        _call({"type": "cosine", "warmup_fraction": 0.01, "warmup_tokens": 1000})


def test_type_foreign_field_raises():
    with pytest.raises(ValueError, match="does not accept field"):
        _call({"type": "cosine", "warmup_fraction": 0.01, "wsd_decay_style": "cosine"})


def test_warmup_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match=r"warmup_fraction must be in \[0, 1\)"):
        _call({"type": "cosine", "warmup_fraction": 1.5})


def test_wsd_decay_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match=r"wsd_decay_fraction must be in \(0, 1\]"):
        _call({"type": "wsd", "warmup_fraction": 0.01, "wsd_decay_fraction": 1.5})


def test_bad_wsd_decay_style_raises():
    with pytest.raises(ValueError, match="wsd_decay_style must be one of"):
        _call({"type": "wsd", "warmup_fraction": 0.01, "wsd_decay_fraction": 0.2,
               "wsd_decay_style": "quadratic"})


def test_force_no_warmup_overrides_fraction():
    m = _to_map(_call({"type": "cosine", "warmup_fraction": 0.05, "min_lr_ratio": 0.1},
                      force_no_warmup=True))
    assert m["--lr-warmup-samples"] == "0"
    assert "--lr-warmup-fraction" not in m
```

- [ ] **Step 3.2: Run to verify the validation tests pass**

Run: `pytest tests/unit/test_scheduler.py -q`
Expected: PASS (14 tests total). The resolver from Task 2 already implements all these checks; this task locks them with tests. If any fail, fix the resolver to match the asserted message.

- [ ] **Step 3.3: Commit**

```bash
git add tests/unit/test_scheduler.py
git commit -m "test(scheduler): validation cases for resolver"
```

---

## Task 4: Strip phantom `scheduler:` blocks from training regimes

The regimes own token budget / batch / checkpointing only. Remove their dead `scheduler:` blocks so they no longer collide with the new scheduler group.

**Files:**
- Modify: `configs/training_regime/ablation_20x.yaml`
- Modify: `configs/training_regime/ablation_40x.yaml`
- Modify: `configs/training_regime/final_200x.yaml`
- Modify: `configs/training_regime/final_400x.yaml`
- Modify: `configs/training_regime/final_wsd_decay_only.yaml`

- [ ] **Step 4.1: Remove the `scheduler:` block from `ablation_20x.yaml`**

Delete these lines:

```yaml
scheduler:
  type: wsd
  warmup_tokens: 2_000_000_000
  stable_fraction: 0.8
  decay_fraction: 0.2
  peak_lr: 0.01
  min_lr_ratio: 0.1
  decay_shape: "linear"
```

(Keep the `training:` and `checkpointing:` blocks. The `checkpointing.save_stable_stage_final` line stays — it's a checkpointing concern, not scheduler.)

- [ ] **Step 4.2: Remove the identical `scheduler:` block from `ablation_40x.yaml`, `final_200x.yaml`, and `final_400x.yaml`**

Each of these three files has the same 8-line block as Step 4.1 (only the surrounding `tokens_per_param` / `save_every_tokens` / `keep_last` differ). Delete the `scheduler:` block from each.

- [ ] **Step 4.3: Remove the `scheduler:` block from `final_wsd_decay_only.yaml`**

Delete these lines:

```yaml
scheduler:
  type: wsd_decay_only
  # Warmup + stable are skipped — we resume at the end of stable.
  decay_tokens: null                 # set explicitly at launch
  peak_lr: 0.01
  min_lr_ratio: 0.1
  decay_shape: "linear"
```

Keep `training.resume_from_stable_stage` and `training.tokens_per_param: null`. Move `decay_tokens` to the `training:` block (it's a budget concern, consumed in Task 8):

```yaml
training:
  tokens_per_param: null
  global_batch_size_tokens: 4_194_304
  seq_length: 4096
  micro_batch_size: null
  resume_from_stable_stage: true
  decay_tokens: null                 # set explicitly at launch (e.g. decay_tokens=1_200_000_000)
  stable_checkpoint_dir: null        # set at launch: the stable run's checkpoints dir
```

Declaring `stable_checkpoint_dir` (and `decay_tokens`) in the regime is required so that launch-time overrides like `training.stable_checkpoint_dir=/path` are accepted under OmegaConf struct mode (new keys are otherwise rejected).

- [ ] **Step 4.4: Verify config composition still loads (no scheduler key consumed yet)**

Run: `pytest tests/unit/test_megatron_args.py -q`
Expected: PASS. `_training_args` still uses the legacy `training.lr_decay_style` path (replaced in Task 7), so removing the phantom blocks is invisible to it.

- [ ] **Step 4.5: Commit**

```bash
git add configs/training_regime/
git commit -m "refactor(scheduler): strip phantom scheduler blocks from regimes"
```

---

## Task 5: Create the `configs/scheduler/` group and the launch default

**Files:**
- Create: `configs/scheduler/cosine.yaml`
- Create: `configs/scheduler/wsd.yaml`
- Create: `configs/scheduler/constant.yaml`
- Create: `configs/scheduler/linear.yaml`
- Create: `configs/scheduler/inverse_square_root.yaml`
- Create: `configs/scheduler/wsd_decay_only.yaml`
- Modify: `configs/launch/config.yaml`

- [ ] **Step 5.1: Create the six scheduler files**

`configs/scheduler/cosine.yaml`:

```yaml
# @package _global_
# Cosine decay with linear warmup. The de-facto default.
scheduler:
  type: cosine
  warmup_fraction: 0.01      # 1% of total tokens; auto-scales with the budget
  min_lr_ratio: 0.1          # min_lr = 0.1 × optim.lr
```

`configs/scheduler/wsd.yaml`:

```yaml
# @package _global_
# Warmup-Stable-Decay (MiniCPM, arXiv:2404.06395). Stable at peak, then anneal.
scheduler:
  type: wsd
  warmup_fraction: 0.01
  min_lr_ratio: 0.1
  wsd_decay_fraction: 0.2    # final 20% is the anneal tail
  wsd_decay_style: cosine    # exponential | linear | cosine | minus_sqrt
```

`configs/scheduler/constant.yaml`:

```yaml
# @package _global_
# Constant LR after warmup. Used for the "stable" half of a decay-only run.
scheduler:
  type: constant
  warmup_fraction: 0.01
```

`configs/scheduler/linear.yaml`:

```yaml
# @package _global_
scheduler:
  type: linear
  warmup_fraction: 0.01
  min_lr_ratio: 0.1
```

`configs/scheduler/inverse_square_root.yaml`:

```yaml
# @package _global_
scheduler:
  type: inverse-square-root
  warmup_fraction: 0.01
  min_lr_ratio: 0.1
```

`configs/scheduler/wsd_decay_only.yaml`:

```yaml
# @package _global_
# Decay-only resume: no warmup, the entire run is the WSD anneal. Compose with
# training_regime=final_wsd_decay_only (which sets resume_from_stable_stage).
scheduler:
  type: wsd
  warmup_fraction: 0.0
  min_lr_ratio: 0.1
  wsd_decay_fraction: 1.0    # whole run is anneal
  wsd_decay_style: cosine
```

- [ ] **Step 5.2: Add the scheduler group to the launch defaults**

In `configs/launch/config.yaml`, add `- scheduler: cosine` to the `defaults:` list, after `base/scale`:

```yaml
defaults:
  - base/family: qwen3
  - base/scale: 1_2b
  - scheduler: cosine
  - experiment: champion
  - training_regime: ablation_20x
  - cluster: b200_de
  - data: nemotron_cc_v2_llama31_8b
  - _self_
```

- [ ] **Step 5.3: Verify composition picks up the scheduler block**

Add a quick test to `tests/unit/test_megatron_args.py` (helpers `_parse_overrides`, `_args_to_map` already imported at top of that file):

```python
def test_scheduler_defaults_to_cosine_block():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    assert cfg.scheduler.type == "cosine"
    assert float(cfg.scheduler.warmup_fraction) == 0.01


def test_scheduler_override_selects_wsd():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    assert cfg.scheduler.type == "wsd"
    assert float(cfg.scheduler.wsd_decay_fraction) == 0.2
```

- [ ] **Step 5.4: Run to verify pass**

Run: `pytest tests/unit/test_megatron_args.py -k scheduler -q`
Expected: PASS (2 tests). If `_parse_overrides` cannot find the `scheduler` group, confirm the directory is `configs/scheduler/` and each file begins with `# @package _global_`.

- [ ] **Step 5.5: Commit**

```bash
git add configs/scheduler/ configs/launch/config.yaml tests/unit/test_megatron_args.py
git commit -m "feat(scheduler): add scheduler config group and launch default"
```

---

## Task 6: Wire the resolver into `_training_args`

Replace the hardcoded warmup / decay-style / min-lr lines with one resolver call, preserving nGPT's no-warmup behavior.

**Files:**
- Modify: `src/utils/megatron_args.py` (`_training_args`, lines ~145-159)
- Test: `tests/unit/test_megatron_args.py` (extend)

- [ ] **Step 6.1: Add a failing end-to-end test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_cosine_scheduler_emits_warmup_fraction_and_min_lr():
    cfg = _parse_overrides(
        [
            "base/family=llama3", "base/scale=300m",
            "scheduler=cosine", "experiment=champion",
            "training_regime=ablation_20x", "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-decay-style"] == "cosine"
    assert m["--lr-warmup-fraction"] == "0.01"
    # champion optim.lr = 1.0e-3, min_lr_ratio = 0.1
    assert m["--min-lr"] == str(1.0e-3 * 0.1)
    assert "--lr-decay-step-ratio" not in m


def test_wsd_scheduler_emits_wsd_flags():
    cfg = _parse_overrides(
        [
            "base/family=llama3", "base/scale=300m",
            "scheduler=wsd", "experiment=champion",
            "training_regime=ablation_20x", "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-decay-style"] == "WSD"
    assert m["--lr-wsd-decay-style"] == "cosine"
    assert "--lr-wsd-decay-samples" in m


def test_ngpt_forces_zero_lr_warmup():
    cfg = _parse_overrides(
        [
            "base/family=llama3", "base/scale=300m",
            "scheduler=cosine", "experiment=arch/ngpt",
            "training_regime=ablation_20x", "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-warmup-samples"] == "0"
    assert "--lr-warmup-fraction" not in m
```

- [ ] **Step 6.2: Run to verify failure**

Run: `pytest tests/unit/test_megatron_args.py -k "scheduler or ngpt" -q`
Expected: FAIL — current `_training_args` emits `--lr-warmup-samples` (not `--lr-warmup-fraction`) and reads `training.min_lr`/`training.lr_decay_style`, so the cosine/wsd assertions fail.

- [ ] **Step 6.3: Add the import**

At the top of `src/utils/megatron_args.py`, add to the imports:

```python
from src.utils.scheduler import scheduler_args
```

- [ ] **Step 6.4: Replace the LR/warmup/decay-style block in `_training_args`**

In `src/utils/megatron_args.py`, the current block reads:

```python
    args: list[str] = []
    _add(args, "--micro-batch-size", micro_batch_size)
    _add(args, "--global-batch-size", global_batch_size)
    _add(args, "--train-samples", total_tokens // seq_length)
    _add(args, "--lr-decay-samples", total_tokens // seq_length)
    warmup_samples = (
        0
        if bool(cfg.optim.get("ngpt", {}).get("no_warmup", False))
        else max(1, (total_tokens // seq_length) // 500)
    )
    _add(args, "--lr-warmup-samples", warmup_samples)
    _add(args, "--lr", optim.get("lr", optim.get("adam", {}).get("lr", 1.0e-3)))
    _add(args, "--min-lr", training.get("min_lr", 1.0e-5))
    lr_decay_style = str(training.get("lr_decay_style", "cosine"))
    _add(args, "--lr-decay-style", lr_decay_style)
    _add(args, "--clip-grad", training.get("clip_grad", 1.0))
```

Replace it with:

```python
    peak_lr = optim.get("lr", optim.get("adam", {}).get("lr", 1.0e-3))
    force_no_warmup = bool(cfg.optim.get("ngpt", {}).get("no_warmup", False))

    args: list[str] = []
    _add(args, "--micro-batch-size", micro_batch_size)
    _add(args, "--global-batch-size", global_batch_size)
    _add(args, "--train-samples", total_tokens // seq_length)
    _add(args, "--lr-decay-samples", total_tokens // seq_length)
    _add(args, "--lr", peak_lr)
    args.extend(
        scheduler_args(
            cfg.scheduler,
            peak_lr=float(peak_lr),
            total_tokens=total_tokens,
            seq_length=seq_length,
            force_no_warmup=force_no_warmup,
        )
    )
    _add(args, "--clip-grad", training.get("clip_grad", 1.0))
```

(The `--weight-decay`, `--bf16`, `--cross-entropy-loss-fusion`, `--calculate-per-token-loss`, and `return args` lines below stay unchanged.)

- [ ] **Step 6.5: Run to verify pass**

Run: `pytest tests/unit/test_megatron_args.py -q`
Expected: PASS (all, including the three new ones). The pre-existing `test_llama3_adam_args_*` test still passes — it asserts model/data/optimizer flags, not LR.

- [ ] **Step 6.6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(scheduler): route _training_args through the scheduler resolver"
```

---

## Task 7: Decay-only resume flags

When `training.resume_from_stable_stage` is set, point `--load` at the stable checkpoint and add `--finetune --override-opt-param-scheduler`; derive `total_tokens` from `decay_tokens`.

**Files:**
- Modify: `src/utils/megatron_args.py` (`_training_args` total_tokens; `_logging_args` load/finetune)
- Test: `tests/unit/test_megatron_args.py` (extend)

- [ ] **Step 7.1: Add a failing decay-only test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_decay_only_resume_emits_finetune_and_override():
    cfg = _parse_overrides(
        [
            "base/family=llama3", "base/scale=300m",
            "scheduler=wsd_decay_only",
            "experiment=champion",
            "training_regime=final_wsd_decay_only",
            "cluster=h800_cn",
            "training.decay_tokens=1200000000",
            "training.stable_checkpoint_dir=/tmp/stable_ckpt",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--finetune"] is True
    assert m["--override-opt-param-scheduler"] is True
    assert m["--load"] == "/tmp/stable_ckpt"
    assert m["--lr-decay-style"] == "WSD"
    # whole run is the anneal: warmup 0, wsd tail == total decay samples
    assert m["--lr-warmup-fraction"] == "0.0"
    assert m["--lr-wsd-decay-samples"] == str(1_200_000_000 // 4096)
```

- [ ] **Step 7.2: Run to verify failure**

Run: `pytest tests/unit/test_megatron_args.py::test_decay_only_resume_emits_finetune_and_override -q`
Expected: FAIL — `--finetune` absent, and `total_tokens` computation hits `int(None)` for `tokens_per_param: null` (raises `TypeError`).

- [ ] **Step 7.3: Make `total_tokens` decay-only-aware in `_training_args`**

In `src/utils/megatron_args.py`, replace the `total_tokens` computation:

```python
    total_tokens = int(training.get("total_tokens", 0)) or (
        int(training.get("tokens_per_param", 20)) * int(cfg.base.non_embedding_params)
    )
```

with:

```python
    if bool(training.get("resume_from_stable_stage", False)):
        decay_tokens = training.get("decay_tokens", None)
        if decay_tokens is None:
            raise ValueError(
                "resume_from_stable_stage requires training.decay_tokens "
                "(e.g. training.decay_tokens=1_200_000_000)"
            )
        total_tokens = int(decay_tokens)
    else:
        total_tokens = int(training.get("total_tokens", 0)) or (
            int(training.get("tokens_per_param", 20)) * int(cfg.base.non_embedding_params)
        )
```

- [ ] **Step 7.4: Emit the resume flags in `_logging_args`**

In `src/utils/megatron_args.py`, `_logging_args` currently ends its `_sequence([...])` with the `--load` / `--wandb-*` entries and returns it directly. Change the function to build the list, then append resume flags:

Replace:

```python
def _logging_args(cfg: DictConfig) -> list[str]:
    derived = cfg.get("_derived", {})
    archive = derived.get("run_dir", "runs/pending") if hasattr(derived, "get") else "runs/pending"
    return _sequence(
        [
            ...
            "--save",
            f"{archive}/checkpoints",
            "--load",
            f"{archive}/checkpoints",
            "--wandb-project",
            cfg.wandb.project,
            "--wandb-exp-name",
            f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s{cfg.seed}",
        ]
    )
```

with (note `--load` now uses the stable checkpoint dir when resuming):

```python
def _logging_args(cfg: DictConfig) -> list[str]:
    derived = cfg.get("_derived", {})
    archive = derived.get("run_dir", "runs/pending") if hasattr(derived, "get") else "runs/pending"
    training = cfg.training
    resume = bool(training.get("resume_from_stable_stage", False))
    load_dir = (
        str(training.get("stable_checkpoint_dir"))
        if resume and training.get("stable_checkpoint_dir", None) is not None
        else f"{archive}/checkpoints"
    )
    args = _sequence(
        [
            "--log-interval",
            cfg.training.get("log_interval", 10),
            "--eval-iters",
            cfg.training.get("eval_iters", 32),
            "--eval-interval",
            cfg.training.get("eval_interval", 500),
            "--save-interval",
            cfg.training.get("save_interval", 5000),
            "--log-throughput",
            "--tensorboard-dir",
            f"{archive}/tensorboard",
            "--ckpt-format",
            cfg.training.get("ckpt_format", "torch_dist"),
            "--distributed-timeout-minutes",
            60,
            "--save",
            f"{archive}/checkpoints",
            "--load",
            load_dir,
            "--wandb-project",
            cfg.wandb.project,
            "--wandb-exp-name",
            f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s{cfg.seed}",
        ]
    )
    if resume:
        _add(args, "--finetune")
        _add(args, "--override-opt-param-scheduler")
    return args
```

- [ ] **Step 7.5: Run to verify pass**

Run: `pytest tests/unit/test_megatron_args.py -q`
Expected: PASS (all). The non-resume tests are unaffected because `resume_from_stable_stage` defaults False.

- [ ] **Step 7.6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(scheduler): decay-only resume flags (finetune + override scheduler)"
```

---

## Task 8: Documentation & full-suite verification

**Files:**
- Modify: `configs/scheduler/` (no-op check)
- Run: full unit suite

- [ ] **Step 8.1: Update the champion experiment description if it claims WSD**

`configs/experiments/champion.yaml` line 8 says "WSD schedule". It actually composes with `scheduler=cosine` now. Change the description text from `WSD schedule` to `cosine LR schedule with linear warmup` so the doc matches reality. (Pure comment/description edit — no behavioral change.)

- [ ] **Step 8.2: Run the full unit suite**

Run: `pytest tests/unit/ -q`
Expected: PASS, with the step-decay test gone and the new scheduler tests present. Note any failures unrelated to this work (pre-existing) but do not fix them here.

- [ ] **Step 8.3: Smoke the four train scripts compose correctly (dry, no GPU)**

Run:

```bash
python -c "
from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args
for sched in ['cosine','wsd','constant','linear','inverse_square_root']:
    cfg = _parse_overrides(['base/family=llama3','base/scale=300m',
        f'scheduler={sched}','experiment=champion',
        'training_regime=ablation_20x','cluster=h800_cn'])
    args = build_megatron_args(cfg)
    assert '--lr-decay-style' in args, sched
    print(sched, 'OK')
"
```

Expected: prints `cosine OK` … `inverse_square_root OK`.

- [ ] **Step 8.4: Commit**

```bash
git add configs/experiments/champion.yaml
git commit -m "docs(scheduler): champion description matches cosine default"
```

---

## Done criteria

- `scripts/train_adam.sh llama3 scheduler=wsd` runs WSD (verified via emitted `--lr-decay-style WSD`).
- `scheduler=cosine scheduler.warmup_fraction=0.02` tweaks warmup without editing YAML.
- All 5 regimes are scheduler-free; all 6 scheduler files compose.
- Step-decay legacy is gone; zero patches involved.
- `pytest tests/unit/test_scheduler.py tests/unit/test_megatron_args.py` is green.

## Out of scope (per spec)

- POET dead-code cleanup.
- Single-run force-save decay-only (Option 4b).
- Per-parameter-group / decoupled-embedding LR.
- Multi-GPU validation (LR schedule is rank-invariant; nothing GPU-count-specific here).
