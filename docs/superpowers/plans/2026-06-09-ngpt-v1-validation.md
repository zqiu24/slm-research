# nGPT v1 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the already-merged nGPT v1 in slm-research is numerically faithful to the NVIDIA reference, trains end-to-end on the cluster, and delivers a measurable convergence speedup over a matched AdamW baseline at 600M dense scale.

**Architecture:** nGPT is *already implemented* on `main` (model module, Megatron spec + patches, config, 11 tests). This plan does **not** re-implement it. It (a) unblocks the full CPU test suite via a one-module split so the pure-PyTorch parity oracle no longer drags in Megatron/transformer_engine, (b) establishes an env where the Megatron-dependent tests run, (c) verifies the nGPT and baseline configs differ only by the intended deltas via a CPU dry-run, then (d) hands the user exact cluster commands for a GPU smoke, an nGPT 600M ablation, and a matched-baseline A/B — interpreting each result against explicit gates.

**Tech Stack:** PyTorch, Megatron-LM Core (vendored `third_party/Megatron-LM`), Hydra/OmegaConf configs, `launchers.submit`, pytest, W&B.

**Spec:** [docs/superpowers/specs/2026-06-09-ngpt-v1-validation-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-09-ngpt-v1-validation-design.md)

---

## Division of labor (READ FIRST — non-negotiable)

- **Claude runs:** all of Stage 0 (Tasks 1–4) — CPU-only: a code split, CPU test runs, a dry-run config diff. Claude reports the *actual* command output.
- **User runs:** all of Stage 1–3 (Tasks 5–7) — every GPU/cluster job. This login node has no usable GPU (driver too old). For each GPU task, Claude prints the exact `python -m launchers.submit ...` command and a paste-back checklist, then **STOPS**. Claude never launches a cluster/GPU job. The user runs it and pastes results back; Claude then evaluates the gate.
- **Per landed task:** update [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) (result log) and [NeckariumAI/zqiu/CHANGELOG.md](/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md).

## Environment facts (measured 2026-06-09)

- Login node, `slm_env` venv: `pytest tests/unit/test_ngpt_*.py` → **26 passed, 3 failed, 2 collection errors**. All 5 non-passing fail because `import megatron.core` → `import transformer_engine` raises `OSError: ... libtransformer_engine.so: undefined symbol: cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13`.
- Test venv (model tests, has torch): `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.
- Launcher/dry-run venv (torch-free, for `launchers.submit`): `/var/tmp/zqiu/slmcpu312/bin/python` (per project memory; pass explicit paths, may need `PYTHONPATH=.`).
- `--dry-run` writes `runs/<run_name>/resolved_config.yaml` (fully resolved) + `launch_metadata.json`; it does **not** touch SLURM or Megatron.

## File map

| Path | Change | Responsibility |
|------|--------|----------------|
| `src/model/ngpt/block.py` | **NEW** | Pure-PyTorch `NGPTBlock` + helpers (`_residual_blend`, `_apply_rope`, `_sinusoidal_embeddings`). No Megatron import. The CPU parity oracle. |
| `src/model/ngpt/layer.py` | MODIFY | Keep only `NGPTTransformerLayer(TransformerLayer)`; import the helpers/`NGPTBlock` from `block.py`. Re-export `NGPTBlock` for back-compat. |
| `tests/unit/test_ngpt_layer_block_forward.py` | MODIFY | Import `NGPTBlock` from `src.model.ngpt.block`. |
| `tests/unit/test_ngpt_full_parity.py` | MODIFY | Import `NGPTBlock` from `src.model.ngpt.block`. |
| `scripts/ngpt_config_parity.py` | **NEW** | CPU diagnostic: resolve nGPT + baseline arms via the launcher resolver, diff emitted Megatron args, assert only the intended deltas. |
| `docs/experiments/ngpt.md` | MODIFY | Populate result log across Stages 1–3. |
| `/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md` | MODIFY | Log each landed change. |

No other source files change unless a stage exposes a defect.

---

## Task 1: Split the pure-PyTorch `NGPTBlock` out of `layer.py` (Stage 0b)

**Why:** `src/model/ngpt/layer.py:27` does `from megatron.core.transformer.transformer_layer import TransformerLayer` at module top, and `NGPTTransformerLayer` (line 161) uses it as a **class base** — so it cannot be a deferred/`TYPE_CHECKING` import. The two parity tests only need the pure-PyTorch `NGPTBlock`. Moving `NGPTBlock` and its pure helpers into a Megatron-free module makes the parity oracle importable on any CPU (no transformer_engine), which is exactly what the original plan claimed but the import broke.

**Files:**
- Create: `src/model/ngpt/block.py`
- Modify: `src/model/ngpt/layer.py`
- Modify: `tests/unit/test_ngpt_layer_block_forward.py`
- Modify: `tests/unit/test_ngpt_full_parity.py`

- [ ] **Step 1.1: Confirm the current CPU failure (red)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -q 2>&1 | tail -5
```
Expected: collection **errors** with `OSError: ... libtransformer_engine.so: undefined symbol`.

- [ ] **Step 1.2: Read the exact span to move**

Open [src/model/ngpt/layer.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/layer.py). The pieces to move into `block.py` are everything that does **not** depend on `TransformerLayer`:
- the three helpers `_residual_blend`, `_apply_rope`, `_sinusoidal_embeddings`,
- the entire `class NGPTBlock(nn.Module)` (starts at line 68),
- the imports they need: `torch`, `torch.nn as nn`, `justnorm`, `LearnedScaling`, and (`QKHyperNorm`, `NGPTMLPBody` are only referenced via the spec, not by `NGPTBlock` — leave those in `layer.py`).

Verify `NGPTBlock` does not reference `TransformerLayer`, `QKHyperNorm`, or `NGPTMLPBody`:
```bash
sed -n '68,160p' src/model/ngpt/layer.py | grep -nE "TransformerLayer|QKHyperNorm|NGPTMLPBody" || echo "clean: NGPTBlock is Megatron-free"
```
Expected: `clean: NGPTBlock is Megatron-free`.

- [ ] **Step 1.3: Create `src/model/ngpt/block.py`**

Create the file with the module docstring, the imports the helpers + `NGPTBlock` need, and the moved code. Header:
```python
"""Pure-PyTorch nGPT block — the CPU-runnable parity oracle.

`NGPTBlock` mirrors the NVIDIA reference `Block(use_nGPT=1)` (attention +
suv-scaled SwiGLU + hypersphere residual blend, incl. the reference's
internal bf16 attention cast). It imports **no Megatron** so the parity
tests run on any CPU without transformer_engine. The Megatron-integrated
layer lives in `layer.py` and reuses `_residual_blend` from here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.ngpt.normalize import justnorm
from src.model.ngpt.scaling_params import LearnedScaling
```
Then paste the three helper functions (`_residual_blend`, `_apply_rope`, `_sinusoidal_embeddings`) and the `NGPTBlock` class **verbatim** from `layer.py` (lines ~35–159). Do not change their bodies.

- [ ] **Step 1.4: Trim `layer.py` and re-import from `block.py`**

In [src/model/ngpt/layer.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/layer.py): delete the three helper functions and the `NGPTBlock` class (now in `block.py`), and replace them with an import. The `TransformerLayer` import stays (it is the base class). `NGPTTransformerLayer.forward` uses `_residual_blend`, so import it. Result — the top of `layer.py` should read:
```python
from __future__ import annotations

from typing import Optional

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.ngpt.attention import QKHyperNorm  # noqa: F401  (used via spec)
from src.model.ngpt.block import NGPTBlock, _residual_blend  # re-exported; _residual_blend reused
from src.model.ngpt.mlp import NGPTMLPBody  # noqa: F401  (used via spec)
from src.model.ngpt.scaling_params import LearnedScaling
```
`NGPTBlock` is re-exported (kept importable from `layer.py` for any back-compat caller). Keep `__all__` if the file defines one; otherwise the bare import suffices. Leave `NGPTTransformerLayer` unchanged.

- [ ] **Step 1.5: Point the two parity tests at the new module**

In both [tests/unit/test_ngpt_layer_block_forward.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_block_forward.py) and [tests/unit/test_ngpt_full_parity.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_full_parity.py), change:
```python
from src.model.ngpt.layer import NGPTBlock
```
to:
```python
from src.model.ngpt.block import NGPTBlock
```
(Leave the `normalize` / `scaling_params` / `ngpt_reference` imports in `test_ngpt_full_parity.py` unchanged — they are already Megatron-free.)

- [ ] **Step 1.6: Run the two parity tests on CPU (green)**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -v 2>&1 | tail -15
```
Expected: all tests **pass** (no `transformer_engine` import; full-model logit parity vs the NVIDIA reference holds). If a parity test fails on a numeric tolerance rather than import, STOP — that is a real correctness signal, not an env issue; triage before continuing.

- [ ] **Step 1.7: Confirm no regression in the pure-torch suite**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_normalize.py tests/unit/test_ngpt_scaling_params.py \
  tests/unit/test_ngpt_attention.py tests/unit/test_ngpt_mlp.py \
  tests/unit/test_ngpt_output_scaling.py tests/unit/test_ngpt_optimizer_groups.py \
  tests/unit/test_ngpt_megatron_args.py tests/unit/test_ngpt_patch_registry.py \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -q 2>&1 | tail -5
```
Expected: **28 passed** (the 26 previously-passing + the 2 now-unblocked parity tests). `test_ngpt_layer_spec.py` is still expected to error here (needs Megatron) — that is Task 2.

- [ ] **Step 1.8: Commit**

```bash
git add src/model/ngpt/block.py src/model/ngpt/layer.py \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py
git commit -F - <<'EOF'
refactor(ngpt): split pure-torch NGPTBlock into block.py so parity oracle runs without transformer_engine
EOF
```

---

## Task 2: Establish a Megatron-importable env and get the full 11-file suite green

**Why:** The 3 tests in [test_ngpt_layer_spec.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_spec.py) build a real Megatron `ModuleSpec` and need `import megatron.core` to succeed. `megatron/core/utils.py:66` imports `transformer_engine` unconditionally, so a *loadable* transformer_engine is mandatory; there is no CPU-only bypass. The `slm_env` TE `.so` references a `cublasLt.so.13` symbol the login node's system lib lacks.

**Files:** none (env work + a recorded command). Output is a documented, reproducible invocation.

- [ ] **Step 2.1: Attempt a local fix — point the loader at the venv's bundled cuBLAS**

The undefined symbol means an older `libcublasLt.so.13` is being resolved before the venv's bundled one. Prepend the venv's NVIDIA libs to the loader path and retry the import:
```bash
VENV=/lustre/fast/fast/zqiu/slm_env/.venv
CUBLAS_DIR=$($VENV/bin/python -c "import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))" 2>/dev/null)/lib
echo "cublas dir: $CUBLAS_DIR"
LD_LIBRARY_PATH="$CUBLAS_DIR:$LD_LIBRARY_PATH" $VENV/bin/python -c "import megatron.core; print('megatron import OK')" 2>&1 | tail -3
```
Expected (best case): `megatron import OK`. If so, this `LD_LIBRARY_PATH` prefix is the working incantation — record it.

- [ ] **Step 2.2: If Step 2.1 fails, run the 3 Megatron tests on a cluster node where `slm_env` loads cleanly**

This is a **non-GPU** test run (the tests don't use CUDA compute), but it needs a node whose system CUDA libs satisfy transformer_engine. Hand the user this command to run on such a node (e.g. an `h800_cn` login/compute node), and have them paste back the tail:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_layer_spec.py -v 2>&1 | tail -15
```
Expected: `3 passed`. (Per the division-of-labor rule, if this must run off the login node, Claude hands the command and waits.)

- [ ] **Step 2.3: Run the complete nGPT suite in the working env (green)**

Using whichever invocation worked (Step 2.1 `LD_LIBRARY_PATH` prefix, or the node from Step 2.2):
```bash
cd /lustre/fast/fast/zqiu/slm-research
# example with the Step 2.1 fix:
LD_LIBRARY_PATH="$CUBLAS_DIR:$LD_LIBRARY_PATH" \
  /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_*.py -q 2>&1 | tail -10
```
Expected: **all nGPT tests pass, zero failures** (the full 11 files).

- [ ] **Step 2.4: Record the reproducible command in the lab notebook**

Add a short "How to run the CPU/parity test suite" subsection to [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) capturing the exact venv + `LD_LIBRARY_PATH` (or node) that makes `import megatron.core` succeed, so this is reproducible.

- [ ] **Step 2.5: Commit**

```bash
git add docs/experiments/ngpt.md
git commit -F - <<'EOF'
docs(ngpt): record reproducible env for the Megatron-dependent unit tests
EOF
```

---

## Task 3: CPU config-parity dry-run — confirm nGPT vs baseline differ only by intent (Stage 0c)

**Why:** The A/B verdict is only meaningful if the two arms are matched except for the method. Catch config drift on CPU before any GPU time. The baseline ([experiment=optim/adam](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/adam.yaml)) defaults to GQA (`num_query_groups: 4`); nGPT forces MHA (`num_query_groups == num_attention_heads == 20`). The matched baseline must override `base.model.num_query_groups=20`.

**Files:**
- Create: `scripts/ngpt_config_parity.py`

- [ ] **Step 3.1: Dry-run the nGPT arm (resolves + archives config; no SLURM)**

Run (torch-free launcher venv; add `PYTHONPATH=.` if needed):
```bash
cd /lustre/fast/fast/zqiu/slm-research
/var/tmp/zqiu/slmcpu312/bin/python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0 --dry-run
```
Expected: JSON with `run_name`, `total_tokens` (≈ 12_000_000_000), `parallelism` `{tp:1, ...}`. Note the printed `archive` path; it contains `resolved_config.yaml`.

- [ ] **Step 3.2: Dry-run the matched baseline arm**

```bash
/var/tmp/zqiu/slmcpu312/bin/python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=optim/adam \
  base.model.num_query_groups=20 \
  training_regime=ablation_20x cluster=h800_cn seed=0 --dry-run
```
Expected: same `total_tokens` and `parallelism.tp=1` as the nGPT arm. Note its `archive` path.

- [ ] **Step 3.3: Write the parity script**

Create `scripts/ngpt_config_parity.py` — it loads both archived `resolved_config.yaml` files, flattens them, and prints the key-by-key diff, classifying each delta as EXPECTED or UNEXPECTED against an allow-list:
```python
"""Diff two resolved_config.yaml archives (nGPT vs matched baseline).

Usage:
    python scripts/ngpt_config_parity.py <ngpt_run_dir> <baseline_run_dir>

Prints every differing leaf key. Keys whose dotted path contains any of
the EXPECTED_DELTA substrings are the intended method differences; anything
else is flagged UNEXPECTED and exits non-zero so drift fails loudly.
"""
import sys

from omegaconf import OmegaConf

# Differences that are *supposed* to exist between the two arms: the optimizer
# recipe (nGPT: ngpt_adamw / lr 15e-4 / wd 0 / no-warmup; adam: adamw / lr 1e-3
# / wd 0.1 / cosine-warmup), the experiment identity + patch list, and the
# baseline's num_query_groups override path (it matches nGPT's MHA, so it should
# NOT actually differ — see below). RoPE/SwiGLU/dims/seq/data/seed must match.
EXPECTED_DELTA = (
    "optim",
    "experiment.name", "experiment.kind", "experiment.family",
    "experiment.patches", "experiment.description", "experiment.references",
    "scheduler",            # nGPT zeroes warmup; baseline keeps it
    "_derived",             # run_name / hashes / archive paths differ by construction
    "wandb.name", "wandb.run",
)


def flat(cfg):
    out = {}
    def rec(node, prefix):
        if hasattr(node, "items"):
            for k, v in node.items():
                rec(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, (list, tuple)):
            out[prefix] = list(node)
        else:
            out[prefix] = node
    rec(cfg, "")
    return out


def main():
    ngpt_dir, base_dir = sys.argv[1], sys.argv[2]
    a = flat(OmegaConf.load(f"{ngpt_dir}/resolved_config.yaml"))
    b = flat(OmegaConf.load(f"{base_dir}/resolved_config.yaml"))
    keys = sorted(set(a) | set(b))
    unexpected = []
    for k in keys:
        if a.get(k) != b.get(k):
            tag = "EXPECTED" if any(s in k for s in EXPECTED_DELTA) else "UNEXPECTED"
            print(f"[{tag}] {k}: ngpt={a.get(k)!r}  baseline={b.get(k)!r}")
            if tag == "UNEXPECTED":
                unexpected.append(k)
    if unexpected:
        print(f"\nFAIL: {len(unexpected)} unexpected config delta(s): {unexpected}")
        sys.exit(1)
    print("\nOK: arms differ only by the intended method/recipe deltas.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run the parity script**

```bash
/var/tmp/zqiu/slmcpu312/bin/python scripts/ngpt_config_parity.py <ngpt_run_dir> <baseline_run_dir>
```
Expected: `OK: arms differ only by the intended method/recipe deltas.` Critically, confirm these match across arms (no UNEXPECTED tag): `base.model.num_attention_heads`, `base.model.num_query_groups` (both 20), `hidden_size`, `ffn_hidden_size`, `num_layers`, `seq_length`, `head_dim`, `positional_encoding`, `tie_embeddings`, `seed`, `data.*`, `training.total_tokens`, `training.global_batch_size`, `parallelism.*`.

If `num_query_groups` shows up UNEXPECTED (20 vs 4), the baseline override didn't take — fix Step 3.2's command before proceeding.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/ngpt_config_parity.py
git commit -F - <<'EOF'
feat(ngpt): CPU config-parity check — diff nGPT vs matched baseline resolved configs
EOF
```

---

## Task 4: Stage 0 gate + handoff prep

**Files:** none (verification + notebook).

- [ ] **Step 4.1: Assert the Stage 0 gate**

Confirm all three hold and write them into the notebook result log:
1. Full nGPT suite green in the working env (Task 2.3).
2. The 2 parity tests green on a plain CPU without transformer_engine (Task 1.6).
3. Config-parity script prints `OK` and the architecture keys match (Task 3.4).

- [ ] **Step 4.2: Update CHANGELOG**

Add a `nGPT validation — Stage 0 (correctness)` entry to [/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md](/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md): module split, env incantation, config-parity tool, suite green.

- [ ] **Step 4.3: Commit**

```bash
git add docs/experiments/ngpt.md /lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md
git commit -F - <<'EOF'
docs(ngpt): Stage 0 validation gate green (suite + parity + config diff)
EOF
```

---

## Task 5: GPU smoke (~100–500 steps) — USER RUNS

**Why:** Confirm nGPT trains end-to-end and its bespoke machinery (per-step weight projection, sqk/suv/alpha/sz, sqrt(head_dim) softmax scale) actually fires on real hardware. Driven by [docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md).

**Files:** `docs/experiments/ngpt.md` (record result).

- [ ] **Step 5.1: Claude prints the command and STOPS**

Hand the user:
```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0 \
  +training.total_tokens=2000000000        # cap ~ a few hundred steps for the smoke
```
Claude does not run this. Wait for the user.

- [ ] **Step 5.2: User runs; pastes back the following evidence**

Paste-back checklist (the gate):
- [ ] Rank-0 stdout shows `[nGPT] applied spec + attached sz + registered weight-norm roles` after build.
- [ ] Loss is strictly decreasing across the first ~50 steps; **no NaN/Inf** in the log.
- [ ] After ~10 steps, a sampled projected matrix has row-norms ≈ 1.0. Snippet to run inside the job (or on a saved step-10 checkpoint):
  ```python
  import torch
  w = model.module.decoder.layers[0].self_attention.linear_qkv.weight  # adjust path to actual module tree
  print("row-norm mean/std:", w.float().norm(dim=1).mean().item(), w.float().norm(dim=1).std().item())
  ```
  Expected mean ≈ 1.0.
- [ ] W&B shows separate `lr_groups/decay` vs `lr_groups/no_decay`; the no-decay group's param count ≈ `2*num_layers (alphas) + per-layer sqk/suv + sz`.

- [ ] **Step 5.3: Resolve the `tie_embeddings` correctness question (BLOCKER before Stage 2)**

600M base sets `tie_embeddings: true`, but the NVIDIA reference unties `wte`/`lm_head` and normalizes them as separate matrices. Determine which holds in the run and that it is intended:
```bash
# In the working test env, inspect the role map + tying on a built model, or grep the impl:
grep -nE "tie_embeddings|output_layer|word_embeddings|_NORM_ROLES|share_embeddings" \
  src/patches/ngpt_apply_spec.py src/model/ngpt/output_scaling.py
```
Decide one of:
- The role map normalizes a single shared tensor once (tying is fine) → record as intended.
- nGPT should untie for parity with the paper → add `base.model.tie_embeddings=false` to the nGPT arm (and, for matched A/B, to the baseline arm too) and note it.

Whatever is chosen must be applied identically to the analysis in Task 3's parity check before Stage 2/3 launch.

- [ ] **Step 5.4: Claude evaluates the gate and records the result**

If all checklist items pass: mark Stage 1 green in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md), update CHANGELOG, commit. If any fail: triage with the runbook's "If it fails" table (softmax_scale, spec-swap firing, projection firing, alphas-in-optimizer), fix on a branch, re-smoke.

---

## Task 6: nGPT 600M ablation — USER RUNS

**Why:** Produce the real nGPT convergence curve (the first arm of the A/B).

**Files:** `docs/experiments/ngpt.md`.

- [ ] **Step 6.1: Claude prints the command and STOPS**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0
```
(≈ 12B tokens = 20 tok/param × 600M; ≈ 2.9k steps at gbs 1024 / seq 4096.) Include any `tie_embeddings` override decided in Step 5.3. Claude does not run this.

- [ ] **Step 6.2: User runs; reports W&B run URL + final/periodic val loss**

Gate: run completes without divergence (no NaN; loss curve smooth); final val loss recorded.

- [ ] **Step 6.3: Claude logs the curve**

Record run URL, seed, token budget, and the val-loss-vs-tokens series in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md); update CHANGELOG; commit.

---

## Task 7: Matched baseline + A/B speedup verdict — USER RUNS

**Why:** The validation verdict — does nGPT converge faster than a standard AdamW baseline at matched architecture/data/seed?

**Files:** `docs/experiments/ngpt.md`.

- [ ] **Step 7.1: Claude prints the matched-baseline command and STOPS**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=optim/adam \
  base.model.num_query_groups=20 \
  training_regime=ablation_20x cluster=h800_cn seed=0
```
Same scale / regime / seed / data as Task 6; only the method differs (nGPT machinery + recipe vs AdamW + its recipe). Apply the same `tie_embeddings` decision as the nGPT arm. Claude does not run this.

- [ ] **Step 7.2: User runs; reports the baseline W&B run URL + val-loss series**

Gate: completes at the matched budget; val-loss-vs-tokens recorded.

- [ ] **Step 7.3: Claude computes the speedup verdict**

From the two W&B series (`tokens_seen` aligned by `wandb_metric_normalize`), compute and record in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md):
1. **Loss at equal tokens** — nGPT vs baseline val loss at the final (and a few intermediate) token counts.
2. **Tokens-to-target** — pick a target val loss the slower arm reaches; report tokens each arm needed → the **speedup factor**.
3. **Documented deviation** — note that both arms used slm-research standard mixed precision (the NVIDIA reference's bf16 param storage, which its README says inflates the speedup, was deliberately not replicated).

- [ ] **Step 7.4: Write the verdict + close out**

Write the final validation verdict (speedup factor, or a null/negative result stated plainly) into the lab notebook result log; update CHANGELOG with `nGPT validation complete`; commit.

---

## Self-review notes (author)

- **Spec coverage:** Stage 0 → Tasks 1–4 (0b split = Task 1, 0a env = Task 2, 0c dry-run = Task 3, gate = Task 4); Stage 1 smoke → Task 5; Stage 2 ablation → Task 6; Stage 3 A/B → Task 7. `tie_embeddings` correctness item → Task 5.3. Documented bf16 deviation → Task 7.3. All spec goals/non-goals mapped.
- **No placeholders:** every code/command step shows the actual command and expected output; the one runtime variable is `<ngpt_run_dir>`/`<baseline_run_dir>` (printed by the dry-run in 3.1/3.2) and `seed=0` (a chosen launch value).
- **Type/name consistency:** `NGPTBlock` moves to `src.model.ngpt.block` and is re-exported from `layer.py`; both parity tests updated to the new path; `_residual_blend` is reused by `layer.py` from `block.py` (single implementation). `build_megatron_args(cfg)` and `--dry-run`/`resolved_config.yaml` match the verified launcher surface.
- **Division of labor:** Tasks 1–4 are CPU/Claude; Tasks 5–7 are USER-run GPU jobs where Claude prints-and-stops, per standing rules.
