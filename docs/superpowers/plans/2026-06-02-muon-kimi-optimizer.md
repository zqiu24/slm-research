# muon_kimi Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `muon_kimi` optimizer that runs the user's vendored Kimi/Moonlight Muon (`GaLore/MUON/muon_kimi.py`) inside the slm-research Megatron loop, selectable via `experiment=optim/muon_kimi` on a single GPU.

**Architecture:** Vendor the optimizer verbatim under `src/optim/` (rides the existing `slm_research` editable install — no new packaging). A builder constructs it and wraps it in Megatron's bf16 optimizer; a POET-style patch routes `--slm-optimizer muon_kimi` to that builder. Config + dev launcher mirror the existing `muon_hybrid`/`*_dev.sh` patterns.

**Tech Stack:** PyTorch, Megatron-Core (pinned in `third_party/Megatron-LM`), OmegaConf/Hydra configs, pytest.

**Deviation from spec:** The spec placed the vendored file at `third_party/muon_kimi/`. During planning we found `third_party/` packages are only importable via per-package *editable installs* (`__editable__.poet_torch-*.pth`); a lone file there would need its own `pip install -e`. Vendoring under `src/optim/_kimi_muon.py` instead rides the existing `slm_research` editable install and imports cleanly in tests and at runtime. Everything else matches the spec.

**Test runner:** all pytest/py_compile commands use the slm_env venv:
`PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`

---

## File Structure

- `src/optim/_kimi_muon.py` — **new**, verbatim copy of the GaLore `Muon` + `zeropower_via_newtonschulz5` (+ provenance/MIT header). One responsibility: the optimizer algorithm.
- `src/optim/muon_kimi.py` — **new**, builder `get_megatron_muon_kimi_optimizer(...)`: param split, construct `Muon`, wrap for Megatron, single-GPU guards.
- `src/patches/muon_kimi_optimizer_setup.py` — **new**, routes `--slm-optimizer muon_kimi` → builder (mirrors `poet_optimizer_setup.py`).
- `src/utils/megatron_args.py` — **modify** `_optimizer_args` (add `muon_kimi` branch).
- `launchers/pretrain_gpt_slm.py` — **modify** `--slm-optimizer` choices (add `muon_kimi`).
- `configs/experiments/optim/muon_kimi.yaml` — **new** experiment config.
- `docs/experiments/muon_kimi.md` — **new** (commit-blocking hook requires it).
- `scripts/train_muon_kimi_dev.sh` — **new** dev launcher (clone of `train_muon_dev.sh`).
- `tests/unit/test_muon_kimi.py` — **new** CPU tests for the vendored optimizer.
- `tests/unit/test_megatron_args.py` — **modify** (add muon_kimi arg test).
- `tests/unit/test_train_scripts.py` — **modify** (add dev-script dry-run test).

---

## Task 1: Vendor the Kimi Muon optimizer

**Files:**
- Create: `src/optim/_kimi_muon.py`
- Test: `tests/unit/test_muon_kimi.py`

- [ ] **Step 1: Copy the optimizer verbatim**

```bash
cd /lustre/fast/fast/zqiu/slm-research
cp /lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_kimi.py src/optim/_kimi_muon.py
```

- [ ] **Step 2: Prepend a provenance/attribution header**

Insert these lines at the very top of `src/optim/_kimi_muon.py` (above the existing `import torch`), leaving the body byte-for-byte unchanged:

```python
"""Vendored Kimi/Moonlight-style Muon optimizer (single-process).

Verbatim copy of /lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_kimi.py, which is
adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py (MIT).
Do not edit the algorithm here; integration lives in src/optim/muon_kimi.py.

Single-GPU only: step() performs no torch.distributed collectives.
"""
```

- [ ] **Step 3: Write the failing test**

Create `tests/unit/test_muon_kimi.py`:

```python
"""CPU tests for the vendored Kimi Muon optimizer (src/optim/_kimi_muon.py)."""

import os

# The vendored Newton-Schulz fn is @torch.compile'd; disable dynamo so the CPU
# test runs eagerly and deterministically. Must be set before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402

from src.optim._kimi_muon import Muon, zeropower_via_newtonschulz5  # noqa: E402


def test_muon_routes_2d_to_muon_and_rest_to_adamw():
    lin = torch.nn.Linear(8, 8, bias=True)
    opt = Muon(lr=1e-2, wd=0.0, muon_params=[lin.weight], adamw_params=[lin.bias])
    assert opt.state[lin.weight]["use_muon"] is True
    assert opt.state[lin.bias]["use_muon"] is False


def test_muon_step_updates_both_param_kinds():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8, bias=True)
    opt = Muon(lr=1e-2, wd=0.0, muon_params=[lin.weight], adamw_params=[lin.bias])
    before_w = lin.weight.detach().clone()
    before_b = lin.bias.detach().clone()
    lin(torch.randn(4, 8)).sum().backward()
    opt.step()
    assert not torch.equal(lin.weight, before_w)
    assert not torch.equal(lin.bias, before_b)


def test_newtonschulz_returns_same_shape():
    g = torch.randn(6, 10)
    out = zeropower_via_newtonschulz5(g, steps=5)
    assert out.shape == g.shape
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m pytest tests/unit/test_muon_kimi.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/_kimi_muon.py tests/unit/test_muon_kimi.py
git commit -m "feat(muon): vendor Kimi Muon optimizer from GaLore (single-process)"
```

---

## Task 2: `_optimizer_args` muon_kimi branch

**Files:**
- Modify: `src/utils/megatron_args.py` (add branch in `_optimizer_args`, after the `muon_hybrid` branch at line 208-225)
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_muon_kimi_argv_routes_through_adam_and_sets_muon_knobs():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "muon_kimi",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "muon_momentum": 0.95,
                "muon_use_nesterov": True,
                "muon_num_ns_steps": 5,
                "adam": {"betas": [0.9, 0.95], "eps": 1.0e-8},
            }
        }
    )
    args = _optimizer_args(cfg)
    amap = {args[i]: args[i + 1] for i in range(0, len(args) - 1)}
    assert amap["--optimizer"] == "adam"
    assert amap["--slm-optimizer"] == "muon_kimi"
    assert amap["--muon-momentum"] == "0.95"
    assert amap["--muon-num-ns-steps"] == "5"
    assert amap["--adam-beta1"] == "0.9"
    assert amap["--adam-beta2"] == "0.95"
    assert "--muon-use-nesterov" in args
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m pytest tests/unit/test_megatron_args.py::test_muon_kimi_argv_routes_through_adam_and_sets_muon_knobs -v
```
Expected: FAIL with `ValueError: Unsupported optimizer type 'muon_kimi'`.

- [ ] **Step 3: Add the muon_kimi branch**

In `src/utils/megatron_args.py`, immediately after the `muon_hybrid` branch (the `return _sequence([...])` block ending at line 225), insert:

```python
    if kind == "muon_kimi":
        adam = optim.get("adam", {})
        betas = adam.get("betas", [0.9, 0.95])
        argv = [
            "--optimizer",
            "adam",
            "--slm-optimizer",
            "muon_kimi",
            "--muon-momentum",
            optim.get("muon_momentum", 0.95),
            "--muon-num-ns-steps",
            optim.get("muon_num_ns_steps", 5),
            "--adam-beta1",
            betas[0],
            "--adam-beta2",
            betas[1],
            "--adam-eps",
            adam.get("eps", 1.0e-8),
        ]
        if bool(optim.get("muon_use_nesterov", True)):
            argv.append("--muon-use-nesterov")
        return _sequence(argv)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m pytest tests/unit/test_megatron_args.py::test_muon_kimi_argv_routes_through_adam_and_sets_muon_knobs -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(muon): emit muon_kimi optimizer args (routes via --optimizer adam)"
```

---

## Task 3: Builder `src/optim/muon_kimi.py`

**Files:**
- Create: `src/optim/muon_kimi.py`

- [ ] **Step 1: Write the builder**

Create `src/optim/muon_kimi.py`:

```python
"""Builder for the ``muon_kimi`` optimizer.

Wraps the vendored single-process Kimi Muon (``src/optim/_kimi_muon.py``) in
Megatron's optimizer machinery. Single-GPU dev scope only: raises on
tensor-parallel / distributed-optimizer / fp16, none of which the vendored
optimizer supports. Reached via ``src/patches/muon_kimi_optimizer_setup.py``.
"""

from __future__ import annotations

from typing import Any

import torch


def get_megatron_muon_kimi_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
) -> Any:
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim._kimi_muon import Muon

    if config.use_distributed_optimizer:
        raise ValueError(
            "muon_kimi does not support the distributed optimizer (single-GPU dev only)."
        )
    if config.fp16:
        raise ValueError("muon_kimi does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError(
            "muon_kimi does not support tensor parallelism > 1 (single-GPU dev only)."
        )

    # Param split mirrors the native muon path (third_party/Megatron-LM/
    # megatron/core/optimizer/muon.py:283-302): 2-D non-embedding/output -> Muon,
    # everything else (embeddings, lm_head, norms, biases) -> internal AdamW.
    muon_params: list = []
    adamw_params: list = []
    for model_chunk in model_chunks:
        for _name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

    optimizer = Muon(
        lr=config.lr,
        wd=config.weight_decay,
        muon_params=muon_params,
        momentum=config.muon_momentum,
        nesterov=config.muon_use_nesterov,
        ns_steps=config.muon_num_ns_steps,
        adamw_params=adamw_params,
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
    )

    def init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                if state.get("use_muon", False):
                    state.setdefault("momentum_buffer", torch.zeros_like(p.data))
                else:
                    if "moment1" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(p.data)
                        state["moment2"] = torch.zeros_like(p.data)

    if config.bf16:
        return Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
    return FP32Optimizer(optimizer, config, init_state_fn)
```

- [ ] **Step 2: Verify it compiles and imports**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m py_compile src/optim/muon_kimi.py && echo "compile OK"
$PY -c "import src.optim.muon_kimi as m; print('import OK', hasattr(m, 'get_megatron_muon_kimi_optimizer'))"
```
Expected: `compile OK` then `import OK True`.

- [ ] **Step 3: Commit**

```bash
git add src/optim/muon_kimi.py
git commit -m "feat(muon): add muon_kimi builder (wraps vendored Muon for Megatron)"
```

---

## Task 4: Routing patch `src/patches/muon_kimi_optimizer_setup.py`

**Files:**
- Create: `src/patches/muon_kimi_optimizer_setup.py`

- [ ] **Step 1: Write the patch (mirrors poet_optimizer_setup.py)**

Create `src/patches/muon_kimi_optimizer_setup.py`:

```python
"""Patch: route slm-research ``muon_kimi`` optimizer through Megatron's Adam branch.

slm-research passes ``--optimizer adam --slm-optimizer muon_kimi``; this patch
tags the OptimizerConfig and reroutes the optimizer-builder call to
``src.optim.muon_kimi.get_megatron_muon_kimi_optimizer``. Mirrors
``poet_optimizer_setup``.
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)


@register_patch(name="muon_kimi_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "muon_kimi":
            return config, overrides
        config.slm_optimizer = "muon_kimi"
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "muon_kimi":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.muon_kimi import get_megatron_muon_kimi_optimizer

        return get_megatron_muon_kimi_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
```

- [ ] **Step 2: Verify it compiles and registers**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m py_compile src/patches/muon_kimi_optimizer_setup.py && echo "compile OK"
$PY -c "import src.patches.muon_kimi_optimizer_setup as p; print('registered:', p.apply.__name__)"
```
Expected: `compile OK` then a line confirming import (no exception).

- [ ] **Step 3: Commit**

```bash
git add src/patches/muon_kimi_optimizer_setup.py
git commit -m "feat(muon): route --slm-optimizer muon_kimi to the kimi builder"
```

---

## Task 5: Add `muon_kimi` to the launcher's optimizer choices

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py:27`

- [ ] **Step 1: Edit the choices list**

Change line 27 from:

```python
        "--slm-optimizer", choices=["adamw", "muon", "poet", "ngpt_adamw"], default="adamw"
```
to:
```python
        "--slm-optimizer", choices=["adamw", "muon", "poet", "ngpt_adamw", "muon_kimi"], default="adamw"
```

- [ ] **Step 2: Verify it compiles**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m py_compile launchers/pretrain_gpt_slm.py && echo "compile OK"
```
Expected: `compile OK`.

- [ ] **Step 3: Commit**

```bash
git add launchers/pretrain_gpt_slm.py
git commit -m "feat(muon): accept --slm-optimizer muon_kimi"
```

---

## Task 6: Experiment config + required doc

**Files:**
- Create: `configs/experiments/optim/muon_kimi.yaml`
- Create: `docs/experiments/muon_kimi.md`

- [ ] **Step 1: Write the experiment config**

Create `configs/experiments/optim/muon_kimi.yaml`:

```yaml
# @package _global_
# Muon variant using the vendored Kimi/Moonlight single-process optimizer
# (src/optim/_kimi_muon.py), routed via --slm-optimizer muon_kimi. Single-GPU
# dev only. Unlike muon_hybrid (separate muon/adam LRs), muon_kimi uses one
# optim.lr for both the Muon and internal-AdamW sides; the Muon side scales
# internally by 0.2*sqrt(max(d_out,d_in)).
experiment:
  name: muon_kimi
  family: optim
  description: |
    Kimi/Moonlight Muon (vendored from the GaLore fork) on 2D attn/MLP weights,
    internal AdamW on embeddings, norms, biases, and LM head. Single base LR;
    Muon update orthogonalized via Newton-Schulz and RMS-scaled. Single-GPU dev.
  references:
    - "Moonlight paper arXiv:2502.16982"
    - "Keller Jordan Muon (github.com/KellerJordan/Muon)"
  patches:
    - model_unfuse_linears
    - muon_kimi_optimizer_setup
    - training_log_eta
    - wandb_metric_normalize
  required_capabilities: []

optim:
  type: muon_kimi
  lr: 1.0e-3
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_use_nesterov: true
  muon_num_ns_steps: 5
  adam:
    betas: [0.9, 0.95]
    eps: 1.0e-8

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 2: Write the required experiment doc**

Create `docs/experiments/muon_kimi.md`:

```markdown
# muon_kimi

Muon variant that runs the vendored Kimi/Moonlight single-process optimizer
(`src/optim/_kimi_muon.py`, adapted from KellerJordan/Muon, MIT) instead of the
Megatron-Core / emerging_optimizers `TensorParallelMuon` used by `muon_hybrid`.

- **Scope:** single GPU (DP=TP=PP=1). The builder
  (`src/optim/muon_kimi.py`) raises on tensor parallelism, the distributed
  optimizer, or fp16.
- **Param routing:** 2-D non-embedding/output weights → Muon; embeddings,
  lm_head, norms, biases → internal AdamW.
- **LR:** one `optim.lr` for both sides (the Muon side is RMS-scaled by
  `0.2*sqrt(max(d_out,d_in))` internally), unlike `muon_hybrid`'s split LRs.
- **Wiring:** `--optimizer adam --slm-optimizer muon_kimi`, rerouted by the
  `muon_kimi_optimizer_setup` patch.

Run on the 60m dev model with `scripts/train_muon_kimi_dev.sh`.
```

- [ ] **Step 3: Verify the experiment-doc hook passes**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY tools/check_experiment_docs.py && echo "doc hook OK"
```
Expected: exits 0, `doc hook OK`.

- [ ] **Step 4: Commit**

```bash
git add configs/experiments/optim/muon_kimi.yaml docs/experiments/muon_kimi.md
git commit -m "feat(muon): add muon_kimi experiment config + doc"
```

---

## Task 7: Dev launcher script + end-to-end dry-run test

**Files:**
- Create: `scripts/train_muon_kimi_dev.sh`
- Modify: `tests/unit/test_train_scripts.py`

- [ ] **Step 1: Create the dev script (clone of train_muon_dev.sh, experiment swapped)**

Create `scripts/train_muon_kimi_dev.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Dev variant for the vendored Kimi Muon (experiment=optim/muon_kimi): same
# harness as train_muon_dev.sh, single-GPU only. Defaults to the tiny 60m
# llama3 scale; any "$@" override still wins.

# torchtitan is AdamW-only in milestone 1; reject --backend torchtitan here.
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
  llama3)
    FAMILY="llama3"
    DEFAULT_SCALE="60m"            # tiny dev scale; override with base/scale=...
    ;;
  deepseek_v3)
    FAMILY="deepseek_v3"
    DEFAULT_SCALE="deepseek_v3_proxy_small"
    ;;
  *)
    echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2
    exit 2
    ;;
esac

# Inject dev defaults unless overridden on the command line.
USER_SET_SCALE="no"
USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

REGIME_ARGS=()
if [[ "${USER_SET_REGIME}" == "no" ]]; then
  REGIME_ARGS=("training_regime=ablation_40x")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/muon_kimi" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "wandb.project=slm-zeju-dev" \
  "$@"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/train_muon_kimi_dev.sh
```

- [ ] **Step 3: Write the failing dry-run test**

Append to `tests/unit/test_train_scripts.py`:

```python
def test_muon_kimi_dev_script_routes_to_kimi_optimizer():
    proc = _run("train_muon_kimi_dev.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "muon_kimi" in proc.stdout
    assert "slm-zeju-dev" in proc.stdout
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
PATH="/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH" \
  $PY -m pytest tests/unit/test_train_scripts.py::test_muon_kimi_dev_script_routes_to_kimi_optimizer -v
```
Expected: PASS. (The test shells out to `bash scripts/train_muon_kimi_dev.sh ... --dry-run`; the launcher prints a JSON command containing `--slm-optimizer muon_kimi`.)

- [ ] **Step 5: Manual dry-run sanity check**

```bash
PATH="/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH" \
  bash scripts/train_muon_kimi_dev.sh llama3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1 \
  | python -c "import sys,json; r=sys.stdin.read(); o=json.loads(r[r.find('{'):]); a=o['command']; g=lambda f:a[a.index(f)+1]; print('optimizer=',g('--optimizer'),'slm-opt=',g('--slm-optimizer'),'lr=',g('--lr'),'num-layers=',g('--num-layers'),'nesterov=', '--muon-use-nesterov' in a)"
```
Expected: `optimizer= adam slm-opt= muon_kimi lr= 0.001 num-layers= 18 nesterov= True`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_muon_kimi_dev.sh tests/unit/test_train_scripts.py
git commit -m "feat(muon): add train_muon_kimi_dev.sh + dry-run smoke"
```

---

## Task 8: Full suite + handoff

- [ ] **Step 1: Run the affected unit tests**

```bash
PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
$PY -m pytest tests/unit/test_muon_kimi.py tests/unit/test_megatron_args.py tests/unit/test_train_scripts.py -v
```
Expected: all pass (modulo the 2 known pre-existing failures noted in the slm CPU-test memory, which are unrelated to these files).

- [ ] **Step 2: Confirm the GPU run is the user's**

The single-GPU training run (`bash scripts/train_muon_kimi_dev.sh`) is run by the user per the GPU policy. Report the exact command and stop; do not launch training.

---

## Self-Review

**Spec coverage:** vendor (T1) ✓; builder + single-GPU guards + bf16 wrap + init_state_fn (T3) ✓; routing patch (T4) ✓; CLI choice + `_optimizer_args` branch (T2, T5) ✓; experiment config (T6) ✓; dev launcher (T7) ✓; verification via py_compile/import + CPU tests + dry-run (T1–T8) ✓; required experiment doc (T6) ✓ — added beyond the spec because it is a commit-blocking hook. Hyperparameter mapping (lr, wd, momentum, nesterov, ns_steps, adamw_betas/eps) matches the spec table.

**Placeholder scan:** none — every code/file step shows complete content; every Run step has an exact command + expected output.

**Type/name consistency:** builder symbol `get_megatron_muon_kimi_optimizer` is identical in T3 (definition), T4 (call site), and the patch. Vendored module `src/optim/_kimi_muon.py` exports `Muon`/`zeropower_via_newtonschulz5`, used consistently in T1 and T3. `optim.type == "muon_kimi"` consistent across T2/T6. `--slm-optimizer muon_kimi` consistent across T2/T4/T5/T7. Config keys (`muon_momentum`, `muon_use_nesterov`, `muon_num_ns_steps`, `adam.betas`, `adam.eps`, `lr`, `weight_decay`) match between T2's reader, T6's config, and the builder's `config.*` reads.
