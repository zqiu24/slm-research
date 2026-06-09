# nGPT v1 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the already-merged nGPT v1 in slm-research is numerically faithful to the NVIDIA reference (via weight-transfer step parity), trains end-to-end on the cluster, and delivers a measurable convergence speedup over a matched AdamW baseline at 600M dense scale.

**Architecture:** nGPT is *already implemented* on `main` (model module, Megatron spec + patches, config, 11 tests). This plan does **not** re-implement it. It (a) unblocks the full CPU test suite via a one-module split so the pure-PyTorch parity oracle no longer drags in Megatron/transformer_engine, (b) establishes an env where the Megatron-dependent tests run, (c) verifies the nGPT and baseline configs differ only by intent via a CPU dry-run, (d) establishes **reference parity** — a CPU full-model forward+one-step parity plus a 1-GPU single-layer Megatron parity vs the NVIDIA reference — and only then (e) hands the user exact cluster commands for a GPU smoke, an nGPT 600M ablation, and a matched-baseline *speedup* A/B.

**Tech Stack:** PyTorch, Megatron-LM Core (vendored `third_party/Megatron-LM`), Hydra/OmegaConf configs, `launchers.submit`, pytest, W&B.

**Spec:** [docs/superpowers/specs/2026-06-09-ngpt-v1-validation-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-09-ngpt-v1-validation-design.md)

---

## Division of labor (READ FIRST — non-negotiable)

- **Claude runs:** Stage 0 (Tasks 1–4) and Stage 0.5a (Task 5) — CPU-only: a code split, CPU test runs, a dry-run config diff, a CPU full-model step-parity test. Claude reports the *actual* command output.
- **User runs:** Stage 0.5b/c (Task 6, a quick 1-GPU single-layer parity) and Stages 1–3 (Tasks 7–9, cluster training). This login node has no usable GPU (driver too old). For each GPU task, Claude prints the exact command + a paste-back checklist, then **STOPS**. Claude never launches a cluster/GPU job. The user runs it and pastes results back; Claude then evaluates the gate.
- **Per landed task:** update [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) (result log) and [NeckariumAI/zqiu/CHANGELOG.md](/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md).

## Environment facts (measured 2026-06-09)

- Login node, `slm_env` venv: `pytest tests/unit/test_ngpt_*.py` → **26 passed, 3 failed, 2 collection errors**. All 5 non-passing fail because `import megatron.core` → `import transformer_engine` raises `OSError: ... libtransformer_engine.so: undefined symbol: cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13`.
- Test venv (model tests, has torch): `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.
- Launcher/dry-run venv (torch-free, for `launchers.submit`): `/var/tmp/zqiu/slmcpu312/bin/python` (per project memory; pass explicit paths, may need `PYTHONPATH=.`).
- `--dry-run` writes `runs/<run_name>/resolved_config.yaml` (fully resolved) + `launch_metadata.json`; it does **not** touch SLURM or Megatron.
- All our tokenizers are >65k vocab (llama3 128256); nanoGPT stores tokens as `uint16` — relevant only if data-port parity is ever revisited (it is not, in this plan).

## File map

| Path | Change | Responsibility |
|------|--------|----------------|
| `src/model/ngpt/block.py` | **NEW** | Pure-PyTorch `NGPTBlock` + helpers (`_residual_blend`, `_apply_rope`, `_sinusoidal_embeddings`). No Megatron import. The CPU parity oracle. |
| `src/model/ngpt/layer.py` | MODIFY | Keep only `NGPTTransformerLayer(TransformerLayer)`; import the helpers/`NGPTBlock` from `block.py`. Re-export `NGPTBlock` for back-compat. |
| `tests/unit/test_ngpt_layer_block_forward.py` | MODIFY | Import `NGPTBlock` from `src.model.ngpt.block`. |
| `tests/unit/test_ngpt_full_parity.py` | MODIFY | Import `NGPTBlock` from `src.model.ngpt.block`. |
| `scripts/ngpt_config_parity.py` | **NEW** | CPU diagnostic: diff nGPT vs baseline resolved configs; assert only intended deltas. |
| `tests/unit/test_ngpt_step_parity.py` | **NEW** | CPU full-model forward + one AdamW step + one normalization parity vs the reference (reuses `_OurNGPT`/`_copy_ref_to_ours`). |
| `tests/numerics/test_ngpt_megatron_layer_parity.py` | **NEW** | 1-GPU single-layer `NGPTTransformerLayer` forward parity vs reference `Block` (RoPE-matched + RoPE-off). `@pytest.mark.gpu`. |
| `docs/experiments/ngpt.md` | MODIFY | Populate result log across all stages. |
| `/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md` | MODIFY | Log each landed change. |

No other source files change unless a stage exposes a defect.

---

## Task 1: Split the pure-PyTorch `NGPTBlock` out of `layer.py` (Stage 0b)

**Why:** `src/model/ngpt/layer.py:27` does `from megatron.core.transformer.transformer_layer import TransformerLayer` at module top, and `NGPTTransformerLayer` (line 161) uses it as a **class base** — so it cannot be a deferred/`TYPE_CHECKING` import. The two parity tests only need the pure-PyTorch `NGPTBlock`. Moving `NGPTBlock` and its pure helpers into a Megatron-free module makes the parity oracle importable on any CPU (no transformer_engine).

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

- [ ] **Step 1.2: Confirm `NGPTBlock` is Megatron-free**

```bash
sed -n '68,160p' src/model/ngpt/layer.py | grep -nE "TransformerLayer|QKHyperNorm|NGPTMLPBody" || echo "clean: NGPTBlock is Megatron-free"
```
Expected: `clean: NGPTBlock is Megatron-free`. (The three helpers `_residual_blend`, `_apply_rope`, `_sinusoidal_embeddings` and `class NGPTBlock` move; the `TransformerLayer` import and `NGPTTransformerLayer` stay.)

- [ ] **Step 1.3: Create `src/model/ngpt/block.py`**

Header + imports, then the three helper functions and `class NGPTBlock` moved **verbatim** from `layer.py`:
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

# <-- paste _residual_blend, _apply_rope, _sinusoidal_embeddings, class NGPTBlock here, verbatim
```

- [ ] **Step 1.4: Trim `layer.py` and re-import from `block.py`**

Delete the three helpers and `NGPTBlock` from `layer.py`; replace with imports. Resulting top of `layer.py`:
```python
from __future__ import annotations

from typing import Optional

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.ngpt.attention import QKHyperNorm  # noqa: F401  (used via spec)
from src.model.ngpt.block import NGPTBlock, _residual_blend  # re-export; _residual_blend reused by forward
from src.model.ngpt.mlp import NGPTMLPBody  # noqa: F401  (used via spec)
from src.model.ngpt.scaling_params import LearnedScaling
```
Leave `NGPTTransformerLayer` unchanged (it uses `_residual_blend`, now imported).

- [ ] **Step 1.5: Point the two parity tests at the new module**

In both [tests/unit/test_ngpt_layer_block_forward.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_block_forward.py) and [tests/unit/test_ngpt_full_parity.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_full_parity.py) change `from src.model.ngpt.layer import NGPTBlock` → `from src.model.ngpt.block import NGPTBlock`. Leave their other imports unchanged.

- [ ] **Step 1.6: Run the two parity tests on CPU (green)**

```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -v 2>&1 | tail -15
```
Expected: all **pass** (no transformer_engine import). If a parity test fails on a numeric tolerance rather than import, STOP — that is a real correctness signal; triage before continuing.

- [ ] **Step 1.7: Confirm no regression in the pure-torch suite**

```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_normalize.py tests/unit/test_ngpt_scaling_params.py \
  tests/unit/test_ngpt_attention.py tests/unit/test_ngpt_mlp.py \
  tests/unit/test_ngpt_output_scaling.py tests/unit/test_ngpt_optimizer_groups.py \
  tests/unit/test_ngpt_megatron_args.py tests/unit/test_ngpt_patch_registry.py \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -q 2>&1 | tail -5
```
Expected: **28 passed** (26 prior + 2 unblocked). `test_ngpt_layer_spec.py` still errors here (needs Megatron) — that is Task 2.

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

**Why:** The 3 tests in [test_ngpt_layer_spec.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_spec.py) build a real Megatron `ModuleSpec` and need `import megatron.core` to succeed. `megatron/core/utils.py:66` imports `transformer_engine` unconditionally, so a *loadable* TE is mandatory; there is no CPU-only bypass. The `slm_env` TE `.so` references a `cublasLt.so.13` symbol the login node's system lib lacks.

**Files:** none (env work + a recorded command).

- [ ] **Step 2.1: Attempt a local fix — point the loader at the venv's bundled cuBLAS**

```bash
VENV=/lustre/fast/fast/zqiu/slm_env/.venv
CUBLAS_DIR=$($VENV/bin/python -c "import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))" 2>/dev/null)/lib
echo "cublas dir: $CUBLAS_DIR"
LD_LIBRARY_PATH="$CUBLAS_DIR:$LD_LIBRARY_PATH" $VENV/bin/python -c "import megatron.core; print('megatron import OK')" 2>&1 | tail -3
```
Expected (best case): `megatron import OK`. If so, record this `LD_LIBRARY_PATH` prefix.

- [ ] **Step 2.2: If Step 2.1 fails, run the Megatron tests on a node where `slm_env` loads cleanly**

Non-GPU test run, but needs a node whose CUDA libs satisfy transformer_engine. Hand the user:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_layer_spec.py -v 2>&1 | tail -15
```
Expected: `3 passed`. If this must run off the login node, Claude hands the command and waits.

- [ ] **Step 2.3: Run the complete nGPT suite in the working env (green)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
LD_LIBRARY_PATH="$CUBLAS_DIR:$LD_LIBRARY_PATH" \
  /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_*.py -q 2>&1 | tail -10
```
Expected: **all nGPT tests pass, zero failures** (the full 11 files).

- [ ] **Step 2.4: Record the reproducible command in the lab notebook**

Add a "How to run the CPU/parity test suite" subsection to [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) with the exact venv + `LD_LIBRARY_PATH` (or node) that makes `import megatron.core` succeed.

- [ ] **Step 2.5: Commit**

```bash
git add docs/experiments/ngpt.md
git commit -F - <<'EOF'
docs(ngpt): record reproducible env for the Megatron-dependent unit tests
EOF
```

---

## Task 3: CPU config-parity dry-run — confirm nGPT vs baseline differ only by intent (Stage 0c)

**Why:** The A/B speedup verdict is only meaningful if the arms are matched except for the method. Catch config drift on CPU before any GPU time. Baseline ([experiment=optim/adam](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/adam.yaml)) defaults to GQA (`num_query_groups: 4`); nGPT forces MHA (`num_query_groups == num_attention_heads == 20`), so the matched baseline must override `base.model.num_query_groups=20`.

**Files:**
- Create: `scripts/ngpt_config_parity.py`

- [ ] **Step 3.1: Dry-run the nGPT arm**

```bash
cd /lustre/fast/fast/zqiu/slm-research
/var/tmp/zqiu/slmcpu312/bin/python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0 --dry-run
```
Expected: JSON with `total_tokens` ≈ 12_000_000_000, `parallelism.tp=1`. Note the printed `archive` path (contains `resolved_config.yaml`).

- [ ] **Step 3.2: Dry-run the matched baseline arm**

```bash
/var/tmp/zqiu/slmcpu312/bin/python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=optim/adam \
  base.model.num_query_groups=20 \
  training_regime=ablation_20x cluster=h800_cn seed=0 --dry-run
```
Expected: same `total_tokens` and `parallelism.tp=1`. Note its `archive` path.

- [ ] **Step 3.3: Write the parity script**

Create `scripts/ngpt_config_parity.py`:
```python
"""Diff two resolved_config.yaml archives (nGPT vs matched baseline).

Usage:
    python scripts/ngpt_config_parity.py <ngpt_run_dir> <baseline_run_dir>

Prints every differing leaf key. Keys whose dotted path contains any of the
EXPECTED_DELTA substrings are intended method differences; anything else is
flagged UNEXPECTED and exits non-zero so drift fails loudly.
"""
import sys

from omegaconf import OmegaConf

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
    unexpected = []
    for k in sorted(set(a) | set(b)):
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
Expected: `OK: ...`. Confirm these are NOT flagged UNEXPECTED (i.e. they match): `base.model.num_attention_heads`, `base.model.num_query_groups` (both 20), `hidden_size`, `ffn_hidden_size`, `num_layers`, `seq_length`, `head_dim`, `positional_encoding`, `tie_embeddings`, `seed`, `data.*`, `training.total_tokens`, `training.global_batch_size`, `parallelism.*`. If `num_query_groups` is UNEXPECTED (20 vs 4), the baseline override didn't take — fix Step 3.2.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/ngpt_config_parity.py
git commit -F - <<'EOF'
feat(ngpt): CPU config-parity check — diff nGPT vs matched baseline resolved configs
EOF
```

---

## Task 4: Stage 0 gate

**Files:** `docs/experiments/ngpt.md`, CHANGELOG.

- [ ] **Step 4.1: Assert the Stage 0 gate**

Confirm and record in the notebook:
1. Full nGPT suite green in the working env (Task 2.3).
2. The 2 parity tests green on plain CPU without transformer_engine (Task 1.6).
3. Config-parity script prints `OK` and the architecture keys match (Task 3.4).

- [ ] **Step 4.2: Update CHANGELOG + commit**

```bash
git add docs/experiments/ngpt.md /lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md
git commit -F - <<'EOF'
docs(ngpt): Stage 0 validation gate green (suite + parity + config diff)
EOF
```

---

## Task 5: CPU full-model forward + one-step parity vs reference (Stage 0.5a)

**Why:** The decisive correctness check for the nGPT *math*. The existing [test_ngpt_full_parity.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_full_parity.py) only checks forward at init. This extends it through one full training step (CE backward → AdamW step → weight-normalization) on **both** the reference and our model from identical transferred weights, then asserts the post-step weights still match — exercising `_residual_blend`, `QKHyperNorm`/`sqk`, `suv`, `sz`, `normalize_module_matrices`, and the optimizer behavior together. Pure-torch, deterministic, CPU.

**Files:**
- Create: `tests/unit/test_ngpt_step_parity.py`

- [ ] **Step 5.1: Write the test (reuses the proven assembly + transfer)**

Create `tests/unit/test_ngpt_step_parity.py`:
```python
"""CPU forward + one-step parity: our pure-torch nGPT vs the NVIDIA reference.

Reuses the model assembly + weight transfer from test_ngpt_full_parity. After
transferring reference weights and normalizing both sides identically, we run
one CE backward + one AdamW step + one weight-normalization on EACH model and
assert the post-step weights still match. This validates the residual blend,
sqk/suv/sz scaling, the matrix projection, and the optimizer step together.

Tolerances are loose: the reference (and our NGPTBlock) cast attention to bf16
internally, so a small precision gap is expected and is not a correctness bug.
"""
import math

import torch

from src.model.ngpt.normalize import justnorm, normalize_module_matrices
from tests._fixtures.ngpt_reference.model import GPT as RefGPT  # noqa: N811
from tests._fixtures.ngpt_reference.model import GPTConfig
from tests.unit.test_ngpt_full_parity import (
    _OurNGPT,
    _build_role_map,
    _copy_ref_to_ours,
    _DEVICE,
)


def _ref_normalize_matrices(ref, n_layer):
    """Mirror NVIDIA train.py::normalize_matrices using justnorm per role."""
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(justnorm(ref.transformer.wte.weight, dim=1))
        ref.lm_head.weight.copy_(justnorm(ref.lm_head.weight, dim=1))
        for i in range(n_layer):
            b = ref.transformer.h[i]
            b.query.weight.copy_(justnorm(b.query.weight, dim=1))
            b.key.weight.copy_(justnorm(b.key.weight, dim=1))
            b.value.weight.copy_(justnorm(b.value.weight, dim=1))
            b.att_c_proj.weight.copy_(justnorm(b.att_c_proj.weight, dim=0))
            b.c_fc.weight.copy_(justnorm(b.c_fc.weight, dim=1))
            b.mlp_c_proj.weight.copy_(justnorm(b.mlp_c_proj.weight, dim=0))


def _cfg():
    n_embd = 32
    return GPTConfig(
        block_size=16, vocab_size=37, n_layer=2, n_head=4, n_embd=n_embd,
        base_scale=1.0 / math.sqrt(n_embd), use_nGPT=1, dropout=0.0, bias=False,
    )


def test_forward_and_one_step_parity():
    torch.manual_seed(7)
    cfg = _cfg()
    ref = RefGPT(cfg).float().to(_DEVICE)
    ours = _OurNGPT(cfg).to(_DEVICE)
    _copy_ref_to_ours(ref, ours, cfg)

    # Init-normalize both identically (reference does this at train.py:411).
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(ours.wte.weight)
        ref.lm_head.weight.copy_(ours.lm_head.weight)
        for i in range(cfg.n_layer):
            rb, ob = ref.transformer.h[i], ours.blocks[i]
            for name in ("query", "key", "value", "att_c_proj", "c_fc", "mlp_c_proj"):
                getattr(rb, name).weight.copy_(getattr(ob, name).weight.to(getattr(rb, name).weight.dtype))

    idx = torch.randint(0, cfg.vocab_size, (2, 8), device=_DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (2, 8), device=_DEVICE)

    # ---- forward parity ----
    ours_logits = ours(idx)
    _, ref_loss0 = ref(idx, targets=tgt)
    ours_loss0 = torch.nn.functional.cross_entropy(
        ours_logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
    )
    assert abs(ours_loss0.item() - ref_loss0.item()) < 5e-2, (
        f"forward loss parity: ours={ours_loss0.item()} ref={ref_loss0.item()}"
    )

    # ---- one AdamW step + normalization on BOTH ----
    opt_args = dict(lr=15e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    o_opt = torch.optim.AdamW(ours.parameters(), **opt_args)
    r_opt = torch.optim.AdamW(ref.parameters(), **opt_args)

    o_opt.zero_grad(); ours_loss0.backward(); o_opt.step()
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))

    r_opt.zero_grad()
    _, ref_loss = ref(idx, targets=tgt)
    ref_loss.backward(); r_opt.step()
    _ref_normalize_matrices(ref, cfg.n_layer)

    # ---- post-step weight parity on sampled tensors ----
    def _max_abs(a, b):
        return (a.float() - b.float()).abs().max().item()

    q_diff = _max_abs(ours.blocks[0].query.weight, ref.transformer.h[0].query.weight)
    wte_diff = _max_abs(ours.wte.weight, ref.transformer.wte.weight)
    alpha_diff = _max_abs(ours.blocks[0].attn_alpha.param, ref.transformer.h[0].attn_alpha)
    sz_diff = _max_abs(ours.sz.param, ref.sz)
    assert q_diff < 5e-2, f"post-step query weight diff {q_diff}"
    assert wte_diff < 5e-2, f"post-step wte diff {wte_diff}"
    assert alpha_diff < 5e-2, f"post-step attn_alpha diff {alpha_diff}"
    assert sz_diff < 5e-2, f"post-step sz diff {sz_diff}"

    # ---- both projected matrices are unit-norm after normalization ----
    assert torch.allclose(
        ours.blocks[0].query.weight.float().norm(dim=1),
        torch.ones(cfg.n_embd, device=_DEVICE), atol=1e-4,
    )
```

- [ ] **Step 5.2: Run it (CPU)**

```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_step_parity.py -v 2>&1 | tail -20
```
Expected: **passed**. If a `*_diff` assertion fails by a wide margin (e.g. >0.5 or NaN), that is a genuine correctness signal in the corresponding primitive — STOP and triage (sign of `alpha`, wrong normalize dim, missing scaling) before continuing. If it fails only slightly above 5e-2, widen the tolerance and note the bf16-attention precision floor in a comment.

- [ ] **Step 5.3: Commit**

```bash
git add tests/unit/test_ngpt_step_parity.py
git commit -F - <<'EOF'
test(ngpt): CPU forward + one-step parity vs NVIDIA reference (residual blend, scaling, normalization, optimizer)
EOF
```

---

## Task 6: 1-GPU single-layer Megatron parity vs reference (Stage 0.5b/c) — USER RUNS

**Why:** Task 5 validates the pure-torch math, but the *Megatron* `NGPTTransformerLayer` + spec is what actually trains. This builds **one** Megatron nGPT layer on a single GPU, transfers the reference `Block` weights into it, feeds an identical hidden state, and compares outputs — validating the wiring Task 5 cannot: `QKHyperNorm` in the `q_layernorm`/`k_layernorm` slots applied post-RoPE, `softmax_scale = sqrt(head_dim)`, `NGPTMLP`'s `suv` path, and the residual blend inside `NGPTTransformerLayer.forward`. There is no infra to build a full Megatron GPTModel in a test, so a single layer is the tractable decisive unit. **This test is developed against the GPU** — the exact `TransformerConfig` field set and module-tree attribute names are confirmed from the first run's errors.

**Files:**
- Create: `tests/numerics/test_ngpt_megatron_layer_parity.py`

- [ ] **Step 6.1: Claude writes the test scaffold (CPU; cannot run here)**

Create `tests/numerics/test_ngpt_megatron_layer_parity.py`. The scaffold below is concrete but the executor must (a) fill any `TransformerConfig` fields the constructor demands and (b) confirm the qkv-fusion layout on first GPU run:
```python
"""Single-layer parity: Megatron NGPTTransformerLayer vs reference Block.

Runs on 1 GPU. We build ONE nGPT layer from the production spec, transfer a
reference Block's weights into it, feed an identical hidden state, and compare
outputs. We test twice: (A) RoPE OFF on both sides to isolate the nGPT math,
then (B) RoPE matched (interleaved, base 10000) to validate position encoding.
"""
import math

import pytest
import torch

pytestmark = [pytest.mark.gpu]

from tests._fixtures.ngpt_reference.model import Block as RefBlock  # noqa: N811,E402
from tests._fixtures.ngpt_reference.model import GPTConfig  # noqa: E402


def _build_megatron_ngpt_layer(hidden, heads, ffn, base_scale):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads,
        ffn_hidden_size=ffn,
        kv_channels=hidden // heads,
        num_query_groups=heads,        # MHA, matches nGPT
        add_bias_linear=False,
        gated_linear_unit=True,        # SwiGLU
        activation_func=torch.nn.functional.silu,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=False,
        params_dtype=torch.float32,
        recompute_granularity=None,
    )
    # Fields the nGPT layer/spec read off config (normally stamped by the
    # ngpt_apply_spec patch). Set explicitly for the standalone test.
    config.ngpt_base_scale = base_scale
    config.ngpt_alpha_init = 0.05
    config.ngpt_sqk_init = 1.0
    config.ngpt_suv_init = 1.0
    config.softmax_scale = math.sqrt(hidden // heads)  # sqrt(head_dim)

    spec = build_ngpt_layer_spec(config)
    layer = build_module(spec, config=config, layer_number=1).cuda().float()
    return layer, config


def _transfer(ref_block, layer, hidden, heads):
    """Copy reference Block weights into the Megatron layer.

    Reference Block: separate query/key/value/att_c_proj/c_fc/mlp_c_proj +
    sqk/suv/attn_alpha/mlp_alpha. The spec uses a FUSED linear_qkv; the executor
    confirms whether the experiment's unfuse patch applies here. If qkv is fused,
    interleave q/k/v into linear_qkv.weight per Megatron's [head, (q,k,v), dh]
    layout. The exact submodule attribute paths are confirmed from the built
    `layer` on first run (print(layer) to see the tree).
    """
    raise NotImplementedError(
        "Fill in once the built layer's module tree is known (see print(layer))."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs 1 GPU")
def test_megatron_layer_matches_reference_block_no_rope():
    torch.manual_seed(0)
    hidden, heads, ffn = 64, 4, 256
    base_scale = 1.0 / math.sqrt(hidden)
    cfg = GPTConfig(block_size=16, vocab_size=37, n_layer=1, n_head=heads,
                    n_embd=hidden, base_scale=base_scale, use_nGPT=1, dropout=0.0, bias=False)
    ref = RefBlock(cfg, iblock=0).float().cuda()
    layer, mcfg = _build_megatron_ngpt_layer(hidden, heads, ffn, base_scale)
    _transfer(ref, layer, hidden, heads)

    # Identical input. Megatron layer expects (s, b, h); reference expects (b, s, h).
    s, b = 8, 1
    h_sbh = torch.randn(s, b, hidden, device="cuda")
    h_bsh = h_sbh.transpose(0, 1).contiguous()

    # RoPE OFF on both: reference Block applies RoPE unconditionally, so for this
    # variant patch it out (or compare against a no-RoPE reference fork). The
    # executor decides the cleanest way once the tree is known.
    out_m, _ = layer(h_sbh, attention_mask=None, rotary_pos_emb=None)
    out_r = ref(h_bsh).transpose(0, 1)
    diff = (out_m.float() - out_r.float()).abs().max().item()
    assert diff < 5e-2, f"single-layer (no-RoPE) parity diff = {diff}"
```

- [ ] **Step 6.2: Claude prints the run command and STOPS**

Hand the user (1 GPU; the working env from Task 2):
```bash
cd /lustre/fast/fast/zqiu/slm-research
LD_LIBRARY_PATH="$CUBLAS_DIR:$LD_LIBRARY_PATH" \
  /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/numerics/test_ngpt_megatron_layer_parity.py -v -s 2>&1 | tail -40
```
The first run is expected to fail in `_transfer` / config construction. The user pastes back `print(layer)` (the module tree) and any `TransformerConfig` error so Claude can finalize `_transfer` and the missing config fields. Iterate until green.

- [ ] **Step 6.3: Finalize `_transfer` + RoPE-matched variant**

Once the tree is known, Claude fills `_transfer` (mapping ref `query/key/value` → fused or unfused `linear_qkv`, `att_c_proj` → `linear_proj`, `c_fc` → `linear_fc1`, `mlp_c_proj` → `linear_fc2`, `sqk` → the q/k `QKHyperNorm`, `suv` → MLP, `attn_alpha`/`mlp_alpha` → the layer) and adds a second test `..._rope_matched` building Megatron RoPE with `rotary_interleaved=True`, `rotary_base=10000` and passing `rotary_pos_emb` into the layer, comparing to the reference *with* its RoPE.

**Gate (0.5b/0.5c):** the no-RoPE variant matches within tolerance (nGPT math wired correctly); the RoPE-matched variant matches too — OR, if only the no-RoPE variant matches, the RoPE convention difference is documented in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) as a known deviation from the reference recipe.

- [ ] **Step 6.4: Commit (after user confirms green)**

```bash
git add tests/numerics/test_ngpt_megatron_layer_parity.py docs/experiments/ngpt.md
git commit -F - <<'EOF'
test(ngpt): 1-GPU single-layer Megatron parity vs reference Block (RoPE-off + RoPE-matched)
EOF
```

---

## Task 7: GPU smoke (~100–500 steps) — USER RUNS

**Why:** Confirm nGPT trains end-to-end and its bespoke machinery (per-step weight projection, sqk/suv/alpha/sz, sqrt(head_dim) softmax scale) fires on real hardware. Driven by [docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md).

**Files:** `docs/experiments/ngpt.md`.

- [ ] **Step 7.1: Claude prints the command and STOPS**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0 \
  training.tokens_per_param=1        # ~143 steps for the smoke (= 600M tokens / seq 4096 / gbs 1024)
```

> **Cap the smoke via `tokens_per_param`, NOT `total_tokens`.** `launchers/submit.py:resolve_config` (submit.py:153) **unconditionally** recomputes `cfg.training.total_tokens = tokens_per_param * non_embedding_params` unless `resume_from_stable_stage` is set — so a `+training.total_tokens=...` override is silently clobbered. (The 2026-05-25 smoke runbook uses the stale `+training.total_tokens=2000000000` form and should be corrected.) `tokens_per_param=1` → 600M tokens → 146,484 samples → **143 steps**; use `0.7` for ~100 steps.

- [ ] **Step 7.2: User runs; pastes back the evidence (the gate)**

- [ ] Rank-0 stdout shows `[nGPT] applied spec + attached sz + registered weight-norm roles` after build.
- [ ] Loss strictly decreasing across the first ~50 steps; **no NaN/Inf**.
- [ ] After ~10 steps, a sampled projected matrix has row-norms ≈ 1.0:
  ```python
  import torch
  w = model.module.decoder.layers[0].self_attention.linear_qkv.weight  # adjust to actual tree
  print("row-norm mean/std:", w.float().norm(dim=1).mean().item(), w.float().norm(dim=1).std().item())
  ```
  Expected mean ≈ 1.0.
- [ ] W&B shows separate `lr_groups/decay` vs `lr_groups/no_decay`; no-decay group contains sz/sqk/suv/attn_alpha/mlp_alpha.

- [ ] **Step 7.3: Resolve the `tie_embeddings` correctness question (BLOCKER before Stage 2)**

600M base sets `tie_embeddings: true`; the reference unties `wte`/`lm_head` and normalizes them separately. Determine which holds and that it is intended:
```bash
grep -nE "tie_embeddings|output_layer|word_embeddings|_NORM_ROLES|share_embeddings" \
  src/patches/ngpt_apply_spec.py src/model/ngpt/output_scaling.py
```
Decide: tying is fine (single shared tensor normalized once) → record as intended; OR untie for parity → add `base.model.tie_embeddings=false` to **both** arms and re-run Task 3's config diff. Apply identically to Stages 2/3.

- [ ] **Step 7.4: Claude evaluates the gate and records**

All items pass → mark Stage 1 green in the notebook, update CHANGELOG, commit. Any fail → triage with the runbook's "If it fails" table, fix on a branch, re-smoke.

---

## Task 8: nGPT 600M ablation — USER RUNS

**Why:** Produce the real nGPT convergence curve (first arm of the A/B).

**Files:** `docs/experiments/ngpt.md`.

- [ ] **Step 8.1: Claude prints the command and STOPS**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  training_regime=ablation_20x cluster=h800_cn seed=0
```
(≈ 12B tokens = 20 tok/param × 600M; ≈ 2.9k steps at gbs 1024 / seq 4096.) Include any `tie_embeddings` override decided in Step 7.3.

- [ ] **Step 8.2: User runs; reports W&B URL + final/periodic val loss**

Gate: completes without divergence (no NaN; smooth curve); final val loss recorded.

- [ ] **Step 8.3: Claude logs the curve**

Record run URL, seed, token budget, val-loss-vs-tokens series in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md); update CHANGELOG; commit.

---

## Task 9: Matched baseline + A/B speedup verdict — USER RUNS

**Why:** The speedup verdict — does nGPT converge faster than a matched AdamW baseline? (Correctness was already established in Tasks 5–6; this is purely speedup.)

**Files:** `docs/experiments/ngpt.md`.

- [ ] **Step 9.1: Claude prints the matched-baseline command and STOPS**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
  base/family=llama3 base/scale=600m experiment=optim/adam \
  base.model.num_query_groups=20 \
  training_regime=ablation_20x cluster=h800_cn seed=0
```
Same scale / regime / seed / data as Task 8; only the method differs. Apply the same `tie_embeddings` decision as the nGPT arm.

- [ ] **Step 9.2: User runs; reports baseline W&B URL + val-loss series**

Gate: completes at the matched budget; val-loss-vs-tokens recorded.

- [ ] **Step 9.3: Claude computes the speedup verdict**

From the two W&B series (`tokens_seen` aligned by `wandb_metric_normalize`), record in [docs/experiments/ngpt.md](/lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md):
1. **Loss at equal tokens** — nGPT vs baseline val loss at final + a few intermediate token counts.
2. **Tokens-to-target** — pick a target val loss the slower arm reaches; report tokens each arm needed → the **speedup factor**.
3. **Documented deviation** — both arms used slm-research standard mixed precision (the reference's bf16 param storage, which its README says inflates speedup, was deliberately not replicated).

- [ ] **Step 9.4: Write the verdict + close out**

Write the final verdict (speedup factor, or a null/negative result stated plainly) into the notebook; update CHANGELOG with `nGPT validation complete`; commit.

---

## Self-review notes (author)

- **Spec coverage:** Stage 0 → Tasks 1–4; Stage 0.5a (CPU step parity) → Task 5; Stage 0.5b/c (Megatron single-layer + RoPE) → Task 6; Stage 1 smoke → Task 7; Stage 2 ablation → Task 8; Stage 3 speedup A/B → Task 9. `tie_embeddings` correctness item → Task 7.3. bf16 deviation → Task 9.3. uint16/RoPE rationale lives in the spec. All spec goals/non-goals mapped.
- **No placeholders:** every CPU step shows the real command + expected output. Task 6 is explicitly a develop-on-GPU test: the scaffold is concrete, with the two genuinely hardware-dependent unknowns (exact `TransformerConfig` fields, qkv-fusion layout / module-tree paths) called out and resolved from the first GPU run — this is the honest shape of a cross-framework GPU parity test, not hand-waving.
- **Type/name consistency:** `NGPTBlock` moves to `src.model.ngpt.block`, re-exported from `layer.py`; both existing parity tests + the new step-parity test import it from `block`. `_residual_blend` single implementation in `block.py`, reused by `layer.py`. Task 5 reuses `_OurNGPT`/`_copy_ref_to_ours`/`_build_role_map`/`_DEVICE` from `test_ngpt_full_parity` (verified to exist). `build_megatron_args(cfg)`, `--dry-run`/`resolved_config.yaml`, `build_ngpt_layer_spec(config)` match the verified code surface.
- **Division of labor:** Tasks 1–5 are CPU/Claude; Tasks 6–9 are USER-run GPU jobs where Claude prints-and-stops, per standing rules.
