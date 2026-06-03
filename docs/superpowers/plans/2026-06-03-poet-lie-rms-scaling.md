# POET × Pion — W-free RMS Scaling (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pion's **Stage 2 RMS scaling** (W-free variant) to the `q_optimizer=lie_algebra` optimizer: after the element-wise Adam direction `A`, scale by `α = rms_c·√(n_blocks·block_size) / (‖A‖_F + ε)` so the per-plane rotation magnitude is consistent across matrices of any width.

**Architecture:** Stage 1 (element-wise Adam on the autograd skew gradient → direction `A`) already ships in `LieAlgebraMomentum`. This adds Stage 2 entirely **inside the optimizer**: a per-param scalar `α` from `‖A‖_F` and the param's own dimension (`√(n_blocks·block_size) = √d`, read off the `oft_R` shape), then `oft_R = lr·α·A`. No `W`, no merge change, no registry — the merge still exponentiates whatever `oft_R` it's handed. The Pion-faithful `‖A·W‖_F` variant (which needs `W`) is explicitly deferred.

**Tech Stack:** Python, PyTorch, Megatron-Core optimizer wrappers, OmegaConf/Hydra, pytest.

**Spec:** [docs/rms_normalization_poet_interval1.md](/lustre/fast/fast/zqiu/slm-research/docs/rms_normalization_poet_interval1.md)

**Conventions (same as the prior lie plans):**
- Repo root: `/lustre/fast/fast/zqiu/slm-research` (run all commands from here).
- CPU test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.
- **Script dry-run tests (Task 5 only)** need the venv on `PATH`: prefix with `PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH`.
- Commit style: single short conventional-commit sentence, anonymous. Let the pre-commit hook run (it reformats with ruff-format and aborts; just re-stage and re-commit).

---

## Key math (verified on CPU before writing)

Per `oft_R` param (shape `(n_blocks, n_elems)`), after Stage 1 gives the direction
`A = -m/(√v+ε)`:

```
dim_const = √(n_blocks · block_size)            # = √d, from the param shape; blocking-invariant
α = rms_c · dim_const / (‖A‖_F + ε)             # ‖A‖_F = Frobenius norm of the A tensor
oft_R = lr · α · A
```

`block_size = block_size_from_nelems(n_elems)`; `n_blocks·block_size = d` (the side's
feature dim) for any blocking. Why `√(n_blocks·block_size) = √d` and not `√(d·d)=d`:
we normalize the *generator* `A` (a rotation, characterized by its per-plane angle
`θ`, with `‖A‖_F = √d · θ_rms`), so `α` pins the **per-plane rotation angle** to a
constant. Verified property: `‖α·A‖_F = rms_c·√(n_blocks·block_size)` **independent
of the gradient**, so `‖oft_R‖_F = lr·rms_c·√d` — scale-consistent by construction.

Stage 1 (element-wise Adam) is unchanged; this only inserts the `α` scaling between
`A` and the write. With alternating, `α` is computed only for the side being written
(it uses that param's own shape + `A`), so it composes for free.

The `‖A·W‖_F` (Pion-faithful) variant is **out of scope** (it needs `W`, hence the
merge — a separate increment); the spec marks W-free as the default and that as an
ablation.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/optim/poet_lie_momentum.py` | add `rms`/`rms_c` to `LieAlgebraMomentum`; Stage-2 scaling in `step()` | Modify |
| `src/optim/poet.py` (builder) | pass `rms`/`rms_c` from config | Modify |
| `launchers/pretrain_gpt_slm.py` | `--poet-lie-rms` / `--poet-lie-rms-c` args | Modify |
| `src/utils/megatron_args.py` | emit the rms args | Modify |
| `src/patches/poet_optimizer_setup.py` | thread `poet_lie_rms` / `poet_lie_rms_c` | Modify |
| `configs/experiments/optim/poet_lie_rms.yaml` | new experiment (`lie_rms: true`) | Create |
| `docs/experiments/poet_lie_rms.md` | required by pre-commit hook | Create |
| `scripts/train_poet_lie_rms.sh` | launcher script | Create |
| `tests/unit/test_poet_lie_momentum.py` | RMS-scaling optimizer tests | Modify |
| `tests/unit/test_pretrain_gpt_slm.py` | launcher accepts rms args | Modify |
| `tests/unit/test_megatron_args.py` | emission + experiment yaml | Modify |
| `tests/unit/test_train_scripts.py` | dry-run smoke | Modify |

---

## Task 1: RMS scaling in `LieAlgebraMomentum`

**Files:**
- Modify: `src/optim/poet_lie_momentum.py`
- Test: `tests/unit/test_poet_lie_momentum.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_lie_momentum.py`:

```python
def _rms_opt(p, lr, rms, rms_c=0.2, v_mode="elementwise"):
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    return LieAlgebraMomentum(
        [dict(params=[p], use_skew=True, side="out", lr=lr)],
        v_mode=v_mode, rms=rms, rms_c=rms_c,
    )


def test_rms_scaling_matches_reference():
    from src.diag.skew_conditioning import block_size_from_nelems

    torch.manual_seed(0)
    ne, lr, rms_c, b1, b2, eps = 6, 1e-3, 0.2, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    g = p.grad.clone()
    # Stage 1 (elementwise Adam, first step from 0) then Stage 2 (W-free RMS):
    m = (1 - b1) * g
    v = (1 - b2) * (g * g)
    A = -m / (v.sqrt() + eps)
    b = block_size_from_nelems(A.shape[1])
    dim_const = (A.shape[0] * b) ** 0.5
    alpha = rms_c * dim_const / (torch.linalg.norm(A) + eps)
    expected = lr * alpha * A
    _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_rms_makes_oft_R_norm_grad_independent():
    # The whole point: ‖oft_R‖_F = lr·rms_c·√(n_blocks·block_size), regardless of
    # the gradient magnitude (scale consistency).
    from src.diag.skew_conditioning import block_size_from_nelems

    ne, lr, rms_c = 6, 1e-3, 0.2
    b = block_size_from_nelems(ne)
    target = lr * rms_c * (1 * b) ** 0.5
    for scale in (1e-3, 1.0, 1e3):  # wildly different gradient magnitudes
        p = nn.Parameter(torch.zeros(1, ne))
        p.grad = scale * torch.randn(1, ne)
        _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
        assert abs(float(torch.linalg.norm(p.data)) - target) < 1e-6, scale


def test_rms_off_is_unscaled():
    # rms=False reproduces the current oft_R = lr*A (no Stage 2).
    torch.manual_seed(2)
    ne, lr, b1, b2, eps = 6, 1e-3, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    g = p.grad.clone()
    A = -((1 - b1) * g) / (((1 - b2) * (g * g)).sqrt() + eps)
    expected = lr * A
    _rms_opt(p, lr, rms=False).step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_rms_norm_scales_with_sqrt_d_across_sizes():
    # Two different widths, same rms_c: ‖oft_R‖/√(n_blocks·block_size) is equal.
    from src.diag.skew_conditioning import block_size_from_nelems

    lr, rms_c = 1e-3, 0.2
    ratios = []
    for d in (4, 8):
        ne = d * (d - 1) // 2
        p = nn.Parameter(torch.zeros(1, ne))
        p.grad = torch.randn(1, ne)
        _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
        b = block_size_from_nelems(ne)
        ratios.append(float(torch.linalg.norm(p.data)) / (1 * b) ** 0.5)
    assert abs(ratios[0] - ratios[1]) < 1e-7, ratios
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -k rms -v
```
Expected: FAIL — `LieAlgebraMomentum.__init__() got an unexpected keyword argument 'rms'`.

- [ ] **Step 3: Add the import**

In `src/optim/poet_lie_momentum.py`, change the imports at the top:

```python
from __future__ import annotations

import torch

from src.diag.skew_conditioning import block_size_from_nelems
```

- [ ] **Step 4: Add `rms` / `rms_c` to the constructor**

In `LieAlgebraMomentum.__init__`, extend the signature (after `alternate_every`)
and store the new attrs. The full signature + the new lines:

```python
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "elementwise",
        alternating: bool = False,
        alternate_every: int = 1,
        rms: bool = False,
        rms_c: float = 0.2,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        # Alternating single-sided update (§6): write only one side's oft_R per
        # step (out on even, in on odd), accumulating momentum on BOTH sides.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
        # Stage 2 RMS scaling (§2, W-free): alpha = rms_c*sqrt(n_blocks*block_size)/(‖A‖_F+eps).
        self.rms = bool(rms)
        self.rms_c = float(rms_c)
```

(Leave the rest of `__init__` — the `defaults = dict(...)` and `super().__init__` —
unchanged.)

- [ ] **Step 5: Insert the Stage-2 scaling in `step()`**

In the skew branch of `step()`, replace the two lines

```python
                    A = -m / (v.sqrt() + eps)
                    p.add_(A.to(p.dtype), alpha=lr)  # p born at 0 -> p = lr*A
```

with:

```python
                    A = -m / (v.sqrt() + eps)
                    if self.rms:
                        # Stage 2 (W-free): scale the rotation generator so the
                        # per-plane angle is dimension-consistent. dim from shape:
                        # sqrt(n_blocks*block_size) = sqrt(d), blocking-invariant.
                        bsz = block_size_from_nelems(A.shape[1])
                        dim_const = (A.shape[0] * bsz) ** 0.5
                        alpha = self.rms_c * dim_const / (torch.linalg.norm(A) + eps)
                        A = A * alpha
                    p.add_(A.to(p.dtype), alpha=lr)  # p born at 0 -> p = lr*(α)A
```

(The RMS block sits *after* the alternating `continue`, so `α` is computed only for
the side actually written.)

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v
```
Expected: PASS — the 4 new rms tests plus all pre-existing optimizer tests.

- [ ] **Step 7: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/optim/poet_lie_momentum.py tests/unit/test_poet_lie_momentum.py && \
git commit -F - <<'EOF'
feat(poet): W-free Stage-2 RMS scaling in LieAlgebraMomentum (α=c·√(n_blocks·block_size)/‖A‖)
EOF
```
(If the pre-commit ruff-format hook aborts, re-run the `git add ... && git commit` once.)

---

## Task 2: Wire `rms` / `rms_c` through the builder

**Files:**
- Modify: `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`)

- [ ] **Step 1: Pass the config to the optimizer**

In `get_megatron_poet_lie_momentum_optimizer`, the `LieAlgebraMomentum(...)`
construction currently ends with `alternate_every=...`. Add two args right after it:

```python
    optimizer = LieAlgebraMomentum(
        param_groups,
        b1=getattr(config, "poet_lie_b1", 0.9),
        b2=getattr(config, "poet_lie_b2", 0.95),
        eps=getattr(config, "poet_lie_eps", 1e-8),
        v_mode=getattr(config, "poet_lie_v_mode", "elementwise"),
        alternating=getattr(config, "poet_lie_alternating", False),
        alternate_every=getattr(config, "poet_lie_alternate_every", 1),
        rms=getattr(config, "poet_lie_rms", False),
        rms_c=getattr(config, "poet_lie_rms_c", 0.2),
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )
```

Also extend the existing `logger.info(...)` call's format string + args to include
`rms` and `rms_c` (append `, rms=%s, rms_c=%s` to the message and
`getattr(config, "poet_lie_rms", False), getattr(config, "poet_lie_rms_c", 0.2)`
to the args) so the run log shows it.

- [ ] **Step 2: Verify it compiles**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/optim/poet.py && echo OK
```
Expected: `OK`. (The full builder needs Megatron → exercised by the Task 5 dry-run.)

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/optim/poet.py && \
git commit -F - <<'EOF'
feat(poet): pass lie_rms/lie_rms_c into the lie_algebra optimizer builder
EOF
```

---

## Task 3: Launcher args

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py`
- Test: `tests/unit/test_pretrain_gpt_slm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_lie_rms():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--poet", "--poet-lie-rms", "--poet-lie-rms-c", "0.3"]
    )
    assert args.poet_lie_rms is True
    assert args.poet_lie_rms_c == 0.3


def test_add_slm_args_lie_rms_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_lie_rms is False
    assert args.poet_lie_rms_c == 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_rms -v
```
Expected: FAIL — `unrecognized arguments: --poet-lie-rms`.

- [ ] **Step 3: Add the args**

In `launchers/pretrain_gpt_slm.py`, immediately after the
`--poet-lie-alternate-every` argument, add:

```python
    # Stage 2 RMS scaling (§2, W-free): scale the rotation generator so per-plane
    # angle is dimension-consistent. alpha = rms_c*sqrt(n_blocks*block_size)/‖A‖.
    group.add_argument("--poet-lie-rms", action="store_true")
    group.add_argument("--poet-lie-rms-c", type=float, default=0.2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_rms -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py && \
git commit -F - <<'EOF'
feat(poet): register --poet-lie-rms / --poet-lie-rms-c launcher args
EOF
```

---

## Task 4: `megatron_args` emission + `poet_optimizer_setup` threading

**Files:**
- Modify: `src/utils/megatron_args.py`
- Modify: `src/patches/poet_optimizer_setup.py`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_emits_lie_rms():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({
        "block_size": 256, "q_optimizer": "lie_algebra",
        "lie_rms": True, "lie_rms_c": 0.3,
    }))
    assert "--poet-lie-rms" in args
    assert args[args.index("--poet-lie-rms-c") + 1] == "0.3"


def test_poet_argv_omits_lie_rms_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-rms" not in args
    assert args[args.index("--poet-lie-rms-c") + 1] == "0.2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_rms -v
```
Expected: FAIL — `--poet-lie-rms-c` not in args.

- [ ] **Step 3: Emit the args (`megatron_args.py`)**

In the `kind == "poet"` branch, in the `poet_args` list, immediately after the
`"--poet-lie-alternate-every", poet.get("lie_alternate_every", 1),` pair, add:

```python
            "--poet-lie-rms-c",
            poet.get("lie_rms_c", 0.2),
```

And in the conditional store_true section (next to the existing
`--poet-lie-alternating` append), add:

```python
        # store_true: enable Stage 2 RMS scaling (W-free) for q_optimizer=lie_algebra.
        if poet.get("lie_rms", False):
            poet_args.append("--poet-lie-rms")
```

- [ ] **Step 4: Thread the config (`poet_optimizer_setup.py`)**

In `_wrapped_get_config`, immediately after the
`config.poet_lie_alternate_every = ...` line, add:

```python
        config.poet_lie_rms = getattr(args, "poet_lie_rms", False)
        config.poet_lie_rms_c = getattr(args, "poet_lie_rms_c", 0.2)
```

- [ ] **Step 5: Run tests + compile**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_rms -v && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_optimizer_setup.py && echo OK
```
Expected: PASS (2 tests) + `OK`.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/utils/megatron_args.py src/patches/poet_optimizer_setup.py tests/unit/test_megatron_args.py && \
git commit -F - <<'EOF'
feat(poet): emit + thread --poet-lie-rms / --poet-lie-rms-c
EOF
```

---

## Task 5: Experiment config + doc + script

**Files:**
- Create: `configs/experiments/optim/poet_lie_rms.yaml`
- Create: `docs/experiments/poet_lie_rms.md`
- Create: `scripts/train_poet_lie_rms.sh`
- Test: `tests/unit/test_megatron_args.py`, `tests/unit/test_train_scripts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_lie_rms_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_rms.yaml")
    assert cfg.experiment.name == "poet_lie_rms"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.lie_rms is True
    assert cfg.optim.poet.lie_rms_c == 0.2
    assert cfg.optim.poet.lie_v_mode == "elementwise"
```

Append to `tests/unit/test_train_scripts.py`:

```python
def test_poet_lie_rms_script_supports_llama3():
    proc = _run("train_poet_lie_rms.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_algebra" in proc.stdout
    assert "--poet-lie-rms" in proc.stdout
    assert "--poet-lie-rms-c" in proc.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k poet_lie_rms_experiment -v
```
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the experiment config**

Create `configs/experiments/optim/poet_lie_rms.yaml`:

```yaml
# @package _global_
# poet_lie_rms: poet_lie + Stage 2 W-free RMS scaling (§2).
#
# Identical to poet_lie (Lie-algebra momentum, element-wise v, single-step,
# block_count=1, reinit_period=-1) but with lie_rms=true: after the Adam direction
# A, scale the rotation generator by alpha = rms_c*sqrt(n_blocks*block_size)/(‖A‖+eps)
# so the per-plane rotation angle is dimension-consistent across matrices. W-free:
# no W access, no merge change. The Pion-faithful ‖A·W‖ variant is deferred.
experiment:
  name: poet_lie_rms
  family: optim
  description: |
    POET x Pion: element-wise Lie-algebra momentum (Stage 1) + W-free RMS scaling
    (Stage 2). q_optimizer=lie_algebra with lie_rms=true normalizes the rotation
    generator's per-plane angle via alpha=rms_c*sqrt(n_blocks*block_size)/‖A‖. Same
    single-step POET stack as poet_lie (merge_period=1, block_count=1,
    reinit_period=-1, cayley); only lie_rms differs. Ablate vs poet_lie; tune lr/c.
  references:
    - "POET"
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
  lr: 1.0e-3
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
    q_optimizer: lie_algebra
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_rms: true             # Stage 2: W-free RMS scaling
    lie_rms_c: 0.2            # RMS target constant (the one new hyperparameter)
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 4: Create the experiment doc**

Create `docs/experiments/poet_lie_rms.md`:

```markdown
# poet_lie_rms — Lie momentum + W-free RMS scaling (§2 Stage 2)

[`poet_lie`](./poet_lie.md) with Pion's **Stage 2 RMS scaling** turned on
(`optim.poet.lie_rms: true`), per
[docs/rms_normalization_poet_interval1.md](../rms_normalization_poet_interval1.md).

After the element-wise Adam direction `A` (Stage 1), the optimizer scales the
rotation generator by

```
α = rms_c · √(n_blocks·block_size) / (‖A‖_F + ε)
oft_R = lr · α · A
```

so the **per-plane rotation angle** is consistent across matrices of any width
(`√(n_blocks·block_size) = √d`, read off the `oft_R` shape — blocking-invariant).
This is **W-free**: no `W` access, no merge change. Net effect:
`‖oft_R‖_F = lr·rms_c·√d`, independent of the gradient magnitude. `rms_c` is the
single new hyperparameter (the RMS target; Pion uses ~0.2).

Everything else matches `poet_lie` (single-step, `block_count=1`,
`reinit_period=-1`, element-wise `v`, Cayley). Run with
[`scripts/train_poet_lie_rms.sh`](../../scripts/train_poet_lie_rms.sh) or
`experiment=optim/poet_lie_rms`; tune `lr` / `lie_rms_c` (RMS scaling enables
larger LR). Composes with alternating (`optim.poet.lie_alternating=true`).

**Deferred:** the Pion-faithful `‖A·W‖_F` normalization (needs `W`, hence the
merge), low-order Cayley / E2 exp, sharded merge.
```

- [ ] **Step 5: Create the script**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
sed 's#experiment=optim/poet_lie#experiment=optim/poet_lie_rms#' \
    scripts/train_poet_lie.sh > scripts/train_poet_lie_rms.sh && \
chmod +x scripts/train_poet_lie_rms.sh
```

Then replace the header comment block (lines 4–8) of
`scripts/train_poet_lie_rms.sh`:

```bash
# poet_lie_rms variant: same harness as train_poet_lie.sh, but uses
# experiment=optim/poet_lie_rms — POET x Pion Lie-algebra momentum
# (q_optimizer=lie_algebra, element-wise v) WITH Stage 2 W-free RMS scaling
# (lie_rms=true): alpha = rms_c*sqrt(n_blocks*block_size)/‖A‖, no W needed.
# Single-step (merge_period=1), reinit_period=-1, block_count=1. "$@" override wins.
```

Verify the only functional difference is the experiment:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
diff <(grep -v '^#' scripts/train_poet_lie.sh) <(grep -v '^#' scripts/train_poet_lie_rms.sh)
```
Expected: a single hunk changing `experiment=optim/poet_lie` → `experiment=optim/poet_lie_rms`.

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k poet_lie_rms_experiment -v && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_train_scripts.py -k poet_lie_rms -v
```
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add configs/experiments/optim/poet_lie_rms.yaml docs/experiments/poet_lie_rms.md scripts/train_poet_lie_rms.sh tests/unit/test_megatron_args.py tests/unit/test_train_scripts.py && \
git commit -F - <<'EOF'
feat(poet): add poet_lie_rms experiment + script (W-free Stage-2 RMS scaling)
EOF
```

---

## Final verification (after all tasks)

- [ ] **In-process unit tests (Tasks 1–4):**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_momentum.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py -v
```
Expected: all new tests PASS. (Pre-existing reds in `test_megatron_args.py` — the
`--poet-merge-period == "200"` assertion and two `wandb_naming` `-scale` ones —
are unrelated; confirmed red before this work.)

- [ ] **Script dry-run (Task 5) with venv on PATH:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k "poet_lie" -v
```
Expected: `poet_lie`, `poet_lie_alt`, and `poet_lie_rms` script tests all PASS.

- [ ] **Static checks:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile \
  src/optim/poet_lie_momentum.py src/optim/poet.py \
  launchers/pretrain_gpt_slm.py src/utils/megatron_args.py \
  src/patches/poet_optimizer_setup.py && \
ruff check src/optim/poet_lie_momentum.py src/optim/poet.py \
  launchers/pretrain_gpt_slm.py src/utils/megatron_args.py \
  src/patches/poet_optimizer_setup.py
```

- [ ] **GPU run (USER — do not run from the agent):** the 60m dev ablation; RMS
  scaling pins `‖oft_R‖=lr·rms_c·√d`, so the effective magnitude is `lr·rms_c` and
  RMS enables a larger LR — sweep `lr` and `lie_rms_c`:

```bash
codexlog poet_lie_rms      bash scripts/train_poet_lie_rms.sh llama3
codexlog poet_lie_rms_alt  bash scripts/train_poet_lie_rms.sh llama3 optim.poet.lie_alternating=true
```
Watch loss vs `poet_lie` (elementwise, no RMS, reached 3.50) and whether RMS lets a
higher LR converge faster/lower. The skew-side LR still decays via the cosine schedule.

---

## Self-Review Notes (author)

- **Spec coverage:** §2 Stage 2 W-free `α=c√(d_out·d_in)/‖A‖` → implemented with the
  corrected per-side `√(n_blocks·block_size)=√d` constant (Task 1); §3 update order
  (scale at the write, exp in the merge) → Task 1 places α between `A` and the write;
  §4 W-free (no W in optimizer) → Task 1 uses only `‖A‖` + shape; §2/§4 `‖A·W‖`
  faithful variant → explicitly deferred (documented in the yaml/doc). Stage 1
  (element-wise Adam) already shipped.
- **vec-norm vs skew-Frobenius:** `torch.linalg.norm(A)` is the norm of the stored
  `(n_blocks, n_elems)` tensor (the upper-tri vec); the skew-matrix `‖A‖_F` is `√2×`
  that. The `√2` (and the per-plane O(1) constant) fold into `rms_c`, which is tuned —
  documented in §key-math. The tests use `torch.linalg.norm(A)` consistently on both
  sides, so they're exact.
- **Naming consistency:** `rms`/`rms_c` (optimizer), `poet_lie_rms`/`poet_lie_rms_c`
  (config/args), `--poet-lie-rms`/`--poet-lie-rms-c` (CLI), `lie_rms`/`lie_rms_c`
  (yaml) used identically across tasks.
- **Composition:** RMS sits after the alternating `continue`, so α is computed only
  for the written side → composes with alternating with no extra code.
