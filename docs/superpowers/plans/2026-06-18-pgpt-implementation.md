# pgpt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pgpt` — a fork of nGPT whose model drops the per-step weight projection (POET preserves the spectrum), keeps a targeted per-step renorm only for the token embedding + lm_head, and is co-trained with POET.

**Architecture:** A full code fork into a parallel `src/model/pgpt/` namespace plus two new patches (`pgpt_apply_spec`, `pgpt_optimizer_setup`), a new layer spec, a new experiment config, and a `_pgpt_arch_args` arg-emitter. pgpt's *forward* is byte-identical to nGPT; the only behavioral changes are on the optimization side: no per-step all-weight renorm (it's redundant under POET's orthogonal updates), and the surviving embedding/lm_head renorm runs from an optimizer post-step hook instead of a `train_step` wrap (so it composes with POET's `poet_merge_step`).

**Tech Stack:** Python 3.12, PyTorch, Megatron-LM (pinned, `third_party/Megatron-LM`), `poet_torch` (`third_party/poet_torch`), Hydra/OmegaConf configs, pytest.

**Spec:** [docs/superpowers/specs/2026-06-18-pgpt-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-18-pgpt-design.md)

## Global Constraints

- **POET-required.** `pgpt_apply_spec` raises if `args.poet` is false. pgpt is only ever run with POET.
- **TP=1, PP=1, dense.** No MoE, no MLA (inherited nGPT v1 constraints; asserted in the spec builder).
- **`base.model.transformer_impl=local`.** Required by POET (`optim.type=poet`) and pinned in the experiment config.
- **`parallelism.distributed_optimizer=false`.** POET's optimizer builder rejects the distributed optimizer; the dev script passes this last.
- **MHA only.** `num_query_groups == num_attention_heads`.
- **Full activation recompute** (`recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1`).
- **Full fork.** No file under `src/model/pgpt/`, `src/specs/pgpt_layer_spec.py`, or `src/patches/pgpt_*.py` may import from `src.model.ngpt`, `src.specs.ngpt_layer_spec`, or `src.patches.ngpt_*`. Zero runtime dependency on nGPT.
- **Keep `ngpt_*` internal names** in the forked code (`config.ngpt_base_scale`, `model._ngpt_sz`, `model._ngpt_norm_role_map`, the `--ngpt*` CLI flags). Renaming to `pgpt_*` is an explicit non-goal for v1.
- **Test interpreter:** `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` (the base `python` lacks `omegaconf`/`torch`). Referred to below as `$PY`. Set once per shell: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.
- **Commit style:** short conventional-commit subject (`feat(pgpt): …`, `test(pgpt): …`), no AI attribution.

---

## File Structure

**Create:**
- `src/model/pgpt/__init__.py`, `normalize.py`, `scaling_params.py`, `attention.py`, `mlp.py`, `output_scaling.py`, `block.py`, `layer.py` — fork of `src/model/ngpt/` with class renames `NGPT*→PGPT*`.
- `src/specs/pgpt_layer_spec.py` — `build_pgpt_layer_spec`.
- `src/patches/pgpt_apply_spec.py` — build-time wrap (spec swap + config stamp + init normalize + role maps + sz + POET-required assert).
- `src/patches/pgpt_optimizer_setup.py` — cooperative `setup_model_and_optimizer` wrap (no-WD groups + embedding/lm_head renorm hook).
- `configs/experiments/arch/pgpt.yaml`, `scripts/train_pgpt_dev.sh`, `docs/experiments/pgpt.md`.
- `tests/_fixtures/pgpt_reference/` — fork of `tests/_fixtures/ngpt_reference/`.
- `tests/unit/test_pgpt_layer_block_forward.py`, `test_pgpt_layer_spec.py`, `test_pgpt_optimizer_groups.py`, `test_pgpt_renorm_hook.py`, `test_pgpt_patch_registry.py`, `test_pgpt_megatron_args.py`.

**Modify:**
- `src/utils/megatron_args.py` — add `_pgpt_arch_args(cfg)` + one call line (do NOT edit `_ngpt_arch_args`).

---

## Task 1: Fork the pgpt model package + CPU forward parity

**Files:**
- Create: `src/model/pgpt/{__init__,normalize,scaling_params,attention,mlp,output_scaling,block,layer}.py`
- Create: `tests/_fixtures/pgpt_reference/` (copy of `tests/_fixtures/ngpt_reference/`)
- Test: `tests/unit/test_pgpt_layer_block_forward.py`

**Interfaces:**
- Produces: `src.model.pgpt.block.PGPTBlock(hidden_size, num_heads, ffn_hidden_size, base_scale, dtype=torch.bfloat16)`; `src.model.pgpt.normalize.justnorm`, `normalize_module_matrices`; `src.model.pgpt.scaling_params.LearnedScaling`; `src.model.pgpt.attention.QKHyperNorm`; `src.model.pgpt.mlp.PGPTMLP`, `PGPTMLPBody`; `src.model.pgpt.layer.PGPTTransformerLayer`; `src.model.pgpt.output_scaling.attach_sz_scaling`.

- [ ] **Step 1: Copy the nGPT model package and reference fixture**

```bash
cd /lustre/fast/fast/zqiu/slm-research
cp -r src/model/ngpt src/model/pgpt
cp -r tests/_fixtures/ngpt_reference tests/_fixtures/pgpt_reference
```

- [ ] **Step 2: Rewire imports and rename classes inside `src/model/pgpt/` only**

Apply these replacements to every `.py` under `src/model/pgpt/` (and nowhere else):

```bash
cd /lustre/fast/fast/zqiu/slm-research
# imports: point the fork at itself
grep -rl 'src.model.ngpt' src/model/pgpt | xargs sed -i 's/src\.model\.ngpt/src.model.pgpt/g'
# class renames (NGPT-prefixed classes only; QKHyperNorm/LearnedScaling keep names)
grep -rl 'NGPT' src/model/pgpt | xargs sed -i \
  -e 's/\bNGPTBlock\b/PGPTBlock/g' \
  -e 's/\bNGPTTransformerLayer\b/PGPTTransformerLayer/g' \
  -e 's/\bNGPTMLPBody\b/PGPTMLPBody/g' \
  -e 's/\bNGPTMLP\b/PGPTMLP/g'
```

Then open `src/model/pgpt/block.py` and `tests/_fixtures/pgpt_reference/model.py` and confirm no remaining `NGPT` class names (the fixture is the NVIDIA reference and intentionally has no `NGPT*` classes — leave it byte-identical to the nGPT copy). Leave `_ngpt_sz`, `_ngpt_norm_role_map`, `config.ngpt_*` strings untouched (Global Constraints: keep `ngpt_*` internal names).

- [ ] **Step 3: Write the failing forward-parity test**

Create `tests/unit/test_pgpt_layer_block_forward.py`:

```python
"""Pure-PyTorch PGPTBlock parity vs the vendored reference Block (use_nGPT=1).

pgpt's forward is byte-identical to nGPT's, so the same reference oracle applies.
"""

import torch

from src.model.pgpt.block import PGPTBlock
from tests._fixtures.pgpt_reference.model import Block as RefBlock
from tests._fixtures.pgpt_reference.model import GPTConfig

try:
    import flash_attn  # noqa: F401

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


def _ref_config(n_embd=64, n_head=4, vocab_size=100):
    return GPTConfig(
        block_size=32,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=n_head,
        n_embd=n_embd,
        base_scale=1.0 / (n_embd**0.5),
        use_nGPT=1,
        dropout=0.0,
        bias=False,
    )


def test_pgpt_block_matches_reference_at_init():
    torch.manual_seed(123)
    cfg = _ref_config()
    ref = RefBlock(cfg, iblock=0).float().to(_DEVICE)
    ours = PGPTBlock(
        hidden_size=cfg.n_embd,
        num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale,
        dtype=torch.float32,
    ).to(_DEVICE)
    with torch.no_grad():
        ours.query.weight.copy_(ref.query.weight)
        ours.key.weight.copy_(ref.key.weight)
        ours.value.weight.copy_(ref.value.weight)
        ours.att_c_proj.weight.copy_(ref.att_c_proj.weight)
        ours.c_fc.weight.copy_(ref.c_fc.weight)
        ours.mlp_c_proj.weight.copy_(ref.mlp_c_proj.weight)
        ours.sqk.param.copy_(ref.sqk)
        ours.suv.param.copy_(ref.suv)
        ours.attn_alpha.param.copy_(ref.attn_alpha)
        ours.mlp_alpha.param.copy_(ref.mlp_alpha)

    x = torch.randn(1, 8, cfg.n_embd, device=_DEVICE)
    ours.eval()
    ref.eval()
    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x).float()
    assert torch.allclose(
        y_ours, y_ref, atol=2e-3, rtol=2e-3
    ), f"max abs diff = {(y_ours - y_ref).abs().max().item()}"


def test_pgpt_block_residual_is_unit_norm_per_token():
    cfg = _ref_config(n_embd=32, n_head=4)
    blk = PGPTBlock(
        hidden_size=cfg.n_embd,
        num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale,
        dtype=torch.float32,
    )
    x = torch.randn(2, 4, cfg.n_embd)
    y = blk(x)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

- [ ] **Step 4: Run the test, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_layer_block_forward.py -v
```
Expected: 2 passed. (The fork is byte-identical math, so it passes immediately. If it errors on import, a rename in Step 2 was missed — fix and rerun.)

- [ ] **Step 5: Verify the no-nGPT-import constraint**

```bash
! grep -rn 'src\.model\.ngpt\|src\.specs\.ngpt_layer_spec\|src\.patches\.ngpt_' src/model/pgpt && echo "CLEAN: no nGPT imports"
```
Expected: prints `CLEAN: no nGPT imports`.

- [ ] **Step 6: Commit**

```bash
git add src/model/pgpt tests/_fixtures/pgpt_reference tests/unit/test_pgpt_layer_block_forward.py
git commit -m "feat(pgpt): fork nGPT model package + CPU forward parity"
```

---

## Task 2: pgpt layer spec

**Files:**
- Create: `src/specs/pgpt_layer_spec.py`
- Test: `tests/unit/test_pgpt_layer_spec.py`

**Interfaces:**
- Consumes: `src.model.pgpt.attention.QKHyperNorm`, `src.model.pgpt.layer.PGPTTransformerLayer`, `src.model.pgpt.mlp.PGPTMLP`.
- Produces: `build_pgpt_layer_spec(config) -> megatron.core.transformer.spec_utils.ModuleSpec`. Asserts `tensor_model_parallel_size == 1`, no MoE, no MLA. Wires `input_layernorm=IdentityOp`, `pre_mlp_layernorm=IdentityOp`, `self_attn_bda=IdentityFuncOp`, `mlp_bda=IdentityFuncOp`, `q_layernorm`/`k_layernorm=QKHyperNorm`-builder, `mlp=ModuleSpec(module=PGPTMLP)`.

- [ ] **Step 1: Create the spec by forking the nGPT spec**

```bash
cd /lustre/fast/fast/zqiu/slm-research
cp src/specs/ngpt_layer_spec.py src/specs/pgpt_layer_spec.py
sed -i \
  -e 's/src\.model\.ngpt/src.model.pgpt/g' \
  -e 's/\bNGPTTransformerLayer\b/PGPTTransformerLayer/g' \
  -e 's/\bNGPTMLP\b/PGPTMLP/g' \
  -e 's/build_ngpt_layer_spec/build_pgpt_layer_spec/g' \
  src/specs/pgpt_layer_spec.py
```

Then update the module docstring's first line in `src/specs/pgpt_layer_spec.py` from "nGPT transformer layer" to "pgpt transformer layer" (cosmetic; leave the rest).

- [ ] **Step 2: Write the failing spec test**

Create `tests/unit/test_pgpt_layer_spec.py`:

```python
"""pgpt layer-spec structure + v1 guardrails (TP=1, no MoE/MLA)."""

import types

import pytest

from src.specs.pgpt_layer_spec import build_pgpt_layer_spec


def _cfg(**over):
    base = dict(
        tensor_model_parallel_size=1,
        num_moe_experts=None,
        multi_latent_attention=False,
        num_attention_heads=4,
        hidden_size=64,
        ngpt_base_scale=1.0 / 8.0,
        ngpt_sqk_init=1.0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_spec_wires_pgpt_layer_and_identity_norms():
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

    from src.model.pgpt.layer import PGPTTransformerLayer
    from src.model.pgpt.mlp import PGPTMLP

    spec = build_pgpt_layer_spec(_cfg())
    assert spec.module is PGPTTransformerLayer
    sub = spec.submodules
    assert sub.input_layernorm is IdentityOp
    assert sub.pre_mlp_layernorm is IdentityOp
    assert sub.self_attn_bda is IdentityFuncOp
    assert sub.mlp_bda is IdentityFuncOp
    assert sub.mlp.module is PGPTMLP


def test_spec_rejects_tp_gt_1():
    with pytest.raises(AssertionError):
        build_pgpt_layer_spec(_cfg(tensor_model_parallel_size=2))


def test_spec_rejects_moe():
    with pytest.raises(AssertionError):
        build_pgpt_layer_spec(_cfg(num_moe_experts=8))
```

- [ ] **Step 3: Run the test, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_layer_spec.py -v
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/specs/pgpt_layer_spec.py tests/unit/test_pgpt_layer_spec.py
git commit -m "feat(pgpt): add pgpt layer spec"
```

---

## Task 3: `pgpt_apply_spec` patch (build-time wiring + POET-required + post-step role map)

**Files:**
- Create: `src/patches/pgpt_apply_spec.py`
- Test: `tests/unit/test_pgpt_apply_spec.py`

**Interfaces:**
- Consumes: `src.specs.pgpt_layer_spec.build_pgpt_layer_spec`, `src.model.pgpt.output_scaling.attach_sz_scaling`, `src.model.pgpt.normalize.normalize_module_matrices`.
- Produces: registered patch `pgpt_apply_spec` (targets `gpt_builders.gpt_builder`, `megatron.training.arguments.core_transformer_config_from_args`). After build it sets, on each model chunk: `model._ngpt_sz` (via `attach_sz_scaling`), `model._ngpt_norm_role_map` (full role map for the one-shot init normalize), and **`model._pgpt_post_step_norm_role_map`** (subset: only the token-embedding and `output_layer` "rows" params, consumed by Task 4). Module-level helper `_match_post_step_role(name) -> str | None`.

- [ ] **Step 1: Create the patch by forking `ngpt_apply_spec`**

```bash
cd /lustre/fast/fast/zqiu/slm-research
cp src/patches/ngpt_apply_spec.py src/patches/pgpt_apply_spec.py
sed -i \
  -e 's/src\.model\.ngpt/src.model.pgpt/g' \
  -e 's/src\.specs\.ngpt_layer_spec/src.specs.pgpt_layer_spec/g' \
  -e 's/build_ngpt_layer_spec/build_pgpt_layer_spec/g' \
  -e 's/_register_ngpt_norm_roles/_register_pgpt_norm_roles/g' \
  -e 's/name="ngpt_apply_spec"/name="pgpt_apply_spec"/g' \
  src/patches/pgpt_apply_spec.py
```

- [ ] **Step 2: Add the POET-required assert**

In `src/patches/pgpt_apply_spec.py`, inside `_wrapped_builder`, immediately after the `if not getattr(args, "ngpt", False): return _orig_builder(...)` guard, insert:

```python
        if not getattr(args, "poet", False):
            raise RuntimeError(
                "pgpt is POET-required: experiment 'arch/pgpt' must run with "
                "optim.type=poet (so --poet is set). Got args.poet=False. "
                "Use scripts/train_pgpt_dev.sh or set optim.type=poet."
            )
```

- [ ] **Step 3: Add the post-step subset role map**

In `src/patches/pgpt_apply_spec.py`, add this module-level helper next to `_match_role`:

```python
# Subset of the role map re-projected every optimizer step (Task 4): only the two
# sphere matrices POET does not wrap (token embedding + lm_head). Both are "rows".
_POST_STEP_ROLES_BY_SUFFIX: dict[tuple[str, ...], str] = {
    ("embedding", "word_embeddings", "weight"): "rows",
    ("output_layer", "weight"): "rows",
}


def _match_post_step_role(name: str) -> str | None:
    parts = name.split(".")
    for suffix, role in _POST_STEP_ROLES_BY_SUFFIX.items():
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == suffix:
            return role
    return None


def _register_pgpt_post_step_roles(model) -> None:
    role_map = {}
    for name, param in model.named_parameters():
        role = _match_post_step_role(name)
        if role is not None:
            role_map[param] = role
    model._pgpt_post_step_norm_role_map = role_map
```

Then, in `_wrapped_builder`, in the post-build `for m in chunks:` loop, add a call right after `_register_pgpt_norm_roles(m, ...)`:

```python
            _register_pgpt_post_step_roles(m)
```

- [ ] **Step 4: Write the failing test**

Create `tests/unit/test_pgpt_apply_spec.py`:

```python
"""pgpt_apply_spec: post-step role matcher + registration."""

import importlib


def test_post_step_role_matches_embedding_and_lm_head():
    from src.patches.pgpt_apply_spec import _match_post_step_role

    assert _match_post_step_role("embedding.word_embeddings.weight") == "rows"
    assert _match_post_step_role("output_layer.weight") == "rows"
    # per-layer POET-wrapped matrices must NOT match (they are not re-projected)
    assert _match_post_step_role("decoder.layers.0.self_attention.linear_qkv.weight") is None
    assert _match_post_step_role("decoder.layers.0.mlp.linear_fc2.weight") is None


def test_pgpt_apply_spec_registers():
    from src.patches._registry import _reset_for_tests, registered_patches

    _reset_for_tests()
    importlib.import_module("src.patches.pgpt_apply_spec")
    assert "pgpt_apply_spec" in registered_patches()
```

- [ ] **Step 5: Run the test, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_apply_spec.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/patches/pgpt_apply_spec.py tests/unit/test_pgpt_apply_spec.py
git commit -m "feat(pgpt): add pgpt_apply_spec (POET-required, post-step role map)"
```

---

## Task 4: `pgpt_optimizer_setup` patch (no-WD groups + embedding/lm_head renorm hook)

**Files:**
- Create: `src/patches/pgpt_optimizer_setup.py`
- Test: `tests/unit/test_pgpt_optimizer_groups.py`, `tests/unit/test_pgpt_renorm_hook.py`

**Interfaces:**
- Consumes: `src.model.pgpt.normalize.normalize_module_matrices`; `model._pgpt_post_step_norm_role_map` (from Task 3).
- Produces: registered patch `pgpt_optimizer_setup` (`targets=()`, cooperative wrap of `megatron.training.training.setup_model_and_optimizer`). Module helper `classify_pgpt_scaling_params(model) -> list[Parameter]` (the `sqk/suv/attn_alpha/mlp_alpha/_ngpt_sz` `.param`s). Internal `_install_renorm_step(optimizer, role_maps)` that monkey-patches `optimizer.step`.

- [ ] **Step 1: Write the failing classifier test**

Create `tests/unit/test_pgpt_optimizer_groups.py`:

```python
"""pgpt scaling-param classifier (no-WD group membership)."""

import torch.nn as nn

from src.model.pgpt.scaling_params import LearnedScaling
from src.patches.pgpt_optimizer_setup import classify_pgpt_scaling_params


def test_scaling_params_are_classified():
    m = nn.Module()
    m.linear = nn.Linear(8, 8, bias=False)
    m.attn_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.mlp_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.sqk = LearnedScaling((8,), init_value=1.0, init_scaling=1.0 / 2.83)
    m.suv = LearnedScaling((8,), init_value=1.0, init_scaling=1.0)
    m._ngpt_sz = LearnedScaling((100,), init_value=1.0, init_scaling=1.0 / 2.83)

    ids = {id(p) for p in classify_pgpt_scaling_params(m)}
    assert id(m.linear.weight) not in ids
    for p in (m.attn_alpha.param, m.mlp_alpha.param, m.sqk.param, m.suv.param, m._ngpt_sz.param):
        assert id(p) in ids
```

- [ ] **Step 2: Run it, expect FAIL (module missing)**

```bash
$PY -m pytest tests/unit/test_pgpt_optimizer_groups.py -v
```
Expected: FAIL with `ModuleNotFoundError: src.patches.pgpt_optimizer_setup`.

- [ ] **Step 3: Write the patch**

Create `src/patches/pgpt_optimizer_setup.py`:

```python
"""Patch: pgpt optimizer-side setup — no-WD scaling groups + targeted renorm.

pgpt is POET-required, so it must NOT wrap ``get_megatron_optimizer`` (POET's
``poet_optimizer_setup`` owns that). Instead this patch cooperatively wraps
``megatron.training.training.setup_model_and_optimizer`` — the same hook
``wandb_trainable_params`` / ``poet_grad_conditioning`` use — and registers
``targets=()`` so it never raises a PatchConflict and composes regardless of
apply order.

After ``setup_model_and_optimizer`` returns it, when ``args.ngpt`` is set:
  (a) moves the nGPT scaling params (sqk/suv/attn_alpha/mlp_alpha/_ngpt_sz) into a
      zero-weight-decay group (belt-and-suspenders; the pgpt config also sets the
      global weight_decay to 0), and
  (b) installs a per-step L2 re-projection of the two sphere matrices POET does
      NOT wrap (token embedding + lm_head) by monkey-patching ``optimizer.step``.
      The per-layer POET-wrapped matrices are intentionally NOT re-projected —
      POET preserves their spectrum. The set comes from
      ``model._pgpt_post_step_norm_role_map`` (registered by pgpt_apply_spec).

NOTE (inherited from nGPT): the renorm mutates the *model* params via the role
map, exactly like nGPT's ``ngpt_normalize_step``. Under a float16 master-weight
optimizer the fp32 master copy is the source of truth; this behavior matches the
validated nGPT path and is confirmed by the GPU smoke, not introduced here.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)

# Trailing module-name segments holding an nGPT scaling vector as ``.param``.
_SCALING_MODULE_NAMES = frozenset({"sqk", "suv", "attn_alpha", "mlp_alpha", "_ngpt_sz"})


def classify_pgpt_scaling_params(model) -> list:
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] == "param" and parts[-2] in _SCALING_MODULE_NAMES:
            out.append(p)
    return out


def _install_renorm_step(optimizer, role_maps) -> None:
    """Monkey-patch ``optimizer.step`` to re-project embedding+lm_head after each step."""
    from src.model.pgpt.normalize import normalize_module_matrices

    if getattr(optimizer, "_pgpt_renorm_installed", False) or not role_maps:
        return
    orig_step = optimizer.step

    def _step(*a, **kw):
        ret = orig_step(*a, **kw)
        for role_map in role_maps:
            normalize_module_matrices(role_map)
        return ret

    optimizer.step = _step  # type: ignore[assignment]
    optimizer._pgpt_renorm_installed = True


@register_patch(name="pgpt_optimizer_setup", targets=())
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    orig = _mt.setup_model_and_optimizer
    if getattr(orig, "_pgpt_optimizer_setup", False):
        return

    def _wrapped(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig(*args, **kwargs)
        if not getattr(get_args(), "ngpt", False):
            return model, optimizer, opt_param_scheduler

        chunks = model if isinstance(model, list | tuple) else [model]

        # (a) zero-WD for scaling params
        scaling_ids: set[int] = set()
        for m in chunks:
            scaling_ids.update(id(p) for p in classify_pgpt_scaling_params(m))
        inner = getattr(optimizer, "optimizer", None) or optimizer
        for group in getattr(inner, "param_groups", []):
            if any(id(p) in scaling_ids for p in group["params"]):
                group["weight_decay"] = 0.0

        # (b) targeted per-step renorm of embedding + lm_head
        role_maps = [
            rm for rm in (getattr(m, "_pgpt_post_step_norm_role_map", None) for m in chunks) if rm
        ]
        _install_renorm_step(optimizer, role_maps)

        logger.info(
            "[pgpt] optimizer setup: zero-WD scaling groups + embedding/lm_head renorm hook"
        )
        return model, optimizer, opt_param_scheduler

    _wrapped._pgpt_optimizer_setup = True
    _mt.setup_model_and_optimizer = _wrapped
```

- [ ] **Step 4: Run the classifier test, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_optimizer_groups.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Write the renorm-hook behavior test**

Create `tests/unit/test_pgpt_renorm_hook.py`:

```python
"""pgpt renorm hook: embedding+lm_head rows go unit-norm; other matrices untouched."""

import torch
import torch.nn as nn

from src.patches.pgpt_optimizer_setup import _install_renorm_step


class _FakeOpt:
    def __init__(self):
        self.stepped = 0

    def step(self):
        self.stepped += 1


def test_renorm_step_projects_only_role_map_params():
    emb = nn.Parameter(torch.randn(10, 8) * 3.0)   # (vocab, hidden) rows
    head = nn.Parameter(torch.randn(10, 8) * 5.0)
    other = nn.Parameter(torch.randn(8, 8) * 7.0)  # a POET-wrapped matrix: untouched
    other_before = other.detach().clone()

    role_map = {emb: "rows", head: "rows"}
    opt = _FakeOpt()
    _install_renorm_step(opt, [role_map])

    opt.step()

    assert opt.stepped == 1
    assert torch.allclose(emb.data.norm(dim=1), torch.ones(10), atol=1e-5)
    assert torch.allclose(head.data.norm(dim=1), torch.ones(10), atol=1e-5)
    assert torch.equal(other.data, other_before)  # not in the role map -> unchanged


def test_install_is_idempotent():
    p = nn.Parameter(torch.randn(4, 4))
    opt = _FakeOpt()
    _install_renorm_step(opt, [{p: "rows"}])
    _install_renorm_step(opt, [{p: "rows"}])  # second call is a no-op
    opt.step()
    assert opt.stepped == 1
```

- [ ] **Step 6: Run it, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_renorm_hook.py -v
```
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/patches/pgpt_optimizer_setup.py tests/unit/test_pgpt_optimizer_groups.py tests/unit/test_pgpt_renorm_hook.py
git commit -m "feat(pgpt): add pgpt_optimizer_setup (no-WD groups + renorm hook)"
```

---

## Task 5: Patch-set conflict-freedom test (core integration claim)

**Files:**
- Test: `tests/unit/test_pgpt_patch_registry.py`

**Interfaces:**
- Consumes: the full pgpt+POET patch name list.
- Produces: nothing (test only). This is the task that proves pgpt and POET co-register without a `PatchConflict`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pgpt_patch_registry.py`:

```python
"""pgpt + POET patch set: registers without PatchConflict; hash is deterministic."""

import importlib

_PGPT_POET_PATCHES = [
    "model_unfuse_linears",
    "poet_apply_to_model",
    "poet_optimizer_setup",
    "poet_merge_step",
    "pgpt_apply_spec",
    "pgpt_optimizer_setup",
]


def _reload(names):
    from src.patches._registry import _reset_for_tests

    _reset_for_tests()
    for n in names:
        importlib.import_module(f"src.patches.{n}")


def test_pgpt_poet_patches_register_without_conflict():
    _reload(_PGPT_POET_PATCHES)
    from src.patches._registry import registered_patches

    reg = registered_patches()
    for n in _PGPT_POET_PATCHES:
        assert n in reg, f"{n} failed to register"


def test_pgpt_patch_set_hash_is_deterministic():
    from src.patches._registry import patch_set_hash

    _reload(_PGPT_POET_PATCHES)
    h1 = patch_set_hash(_PGPT_POET_PATCHES)
    _reload(_PGPT_POET_PATCHES)
    h2 = patch_set_hash(_PGPT_POET_PATCHES)
    assert h1 == h2 and len(h1) == 16
```

- [ ] **Step 2: Run it, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_patch_registry.py -v
```
Expected: 2 passed. (If `PatchConflict` is raised, a pgpt patch is wrongly declaring an exclusive `target` that a POET patch also owns — re-check that `pgpt_optimizer_setup` uses `targets=()`.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_pgpt_patch_registry.py
git commit -m "test(pgpt): pgpt+POET patch set registers without conflict"
```

---

## Task 6: `_pgpt_arch_args` in `megatron_args.py`

**Files:**
- Modify: `src/utils/megatron_args.py` (add `_pgpt_arch_args` + one call line; do NOT edit `_ngpt_arch_args`)
- Test: `tests/unit/test_pgpt_megatron_args.py`

> **Note:** `src/utils/megatron_args.py` may carry unrelated uncommitted edits at execution time (a `_distributed_optimizer_supported` change). Locate insertion points by **function name**, not line number, and keep this task's diff limited to the additions below.

**Interfaces:**
- Consumes: `cfg.experiment.kind`, `cfg.optim.ngpt.*`.
- Produces: `_pgpt_arch_args(cfg) -> list[str]` emitting `--ngpt` + scaling inits when `kind == "pgpt"`, else `[]`. Called from `build_megatron_args` right after the existing `args.extend(_ngpt_arch_args(cfg))` line.

- [ ] **Step 1: Add the `_pgpt_arch_args` helper**

In `src/utils/megatron_args.py`, immediately **after** the `_ngpt_arch_args` function definition, add:

```python
def _pgpt_arch_args(cfg: DictConfig) -> list[str]:
    """pgpt architecture CLI flags (mirror of _ngpt_arch_args, keyed on kind=='pgpt').

    pgpt reuses nGPT's architecture flags and config fields (the forked pgpt model
    reads config.ngpt_* exactly like nGPT). Emitting them here, keyed on
    experiment.kind=='pgpt' (NOT optim.type, which is 'poet'), keeps the optimizer
    branch free to supply the POET flags. No-op when pgpt is not requested.
    """
    experiment = cfg.get("experiment", {}) or {}
    if str(experiment.get("kind", "")) != "pgpt":
        return []
    ng = (cfg.get("optim", {}) or {}).get("ngpt", {}) or {}
    arch = [
        "--ngpt",
        "--ngpt-alpha-init",
        float(ng.get("alpha_init", 0.05)),
        "--ngpt-sqk-init",
        float(ng.get("sqk_init", 1.0)),
        "--ngpt-suv-init",
        float(ng.get("suv_init", 1.0)),
        "--ngpt-sz-init",
        float(ng.get("sz_init", 1.0)),
    ]
    if bool(ng.get("no_warmup", True)):
        arch.append("--ngpt-no-warmup")
    return _sequence(arch)
```

- [ ] **Step 2: Wire the call**

In `src/utils/megatron_args.py`, find the line `args.extend(_ngpt_arch_args(cfg))` inside `build_megatron_args` and add immediately below it:

```python
    args.extend(_pgpt_arch_args(cfg))
```

- [ ] **Step 3: Write the failing test**

Create `tests/unit/test_pgpt_megatron_args.py`:

```python
"""_pgpt_arch_args emits the nGPT arch flags when experiment.kind == 'pgpt'."""

from omegaconf import OmegaConf

from src.utils.megatron_args import _pgpt_arch_args


def _cfg(kind, **ngpt):
    return OmegaConf.create(
        {"experiment": {"kind": kind}, "optim": {"ngpt": ngpt}}
    )


def test_emits_ngpt_flags_for_pgpt_kind():
    out = _pgpt_arch_args(_cfg("pgpt", alpha_init=0.05, sqk_init=1.0, suv_init=1.0, sz_init=1.0))
    assert "--ngpt" in out
    assert "--ngpt-alpha-init" in out
    assert "--ngpt-no-warmup" in out  # default no_warmup=True


def test_noop_for_non_pgpt_kind():
    assert _pgpt_arch_args(_cfg("ngpt")) == []
    assert _pgpt_arch_args(_cfg("adamw")) == []
```

- [ ] **Step 4: Run it, expect PASS**

```bash
$PY -m pytest tests/unit/test_pgpt_megatron_args.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_pgpt_megatron_args.py
git commit -m "feat(pgpt): emit nGPT arch flags for experiment.kind=pgpt"
```

---

## Task 7: Experiment config + dev script + experiment doc

**Files:**
- Create: `configs/experiments/arch/pgpt.yaml`, `scripts/train_pgpt_dev.sh`, `docs/experiments/pgpt.md`

**Interfaces:**
- Consumes: all patches/specs/args from Tasks 1–6.
- Produces: a resolvable experiment that emits both `--ngpt` and `--poet`. (`docs/experiments/pgpt.md` is REQUIRED by the pre-commit hook "Every experiment YAML has a matching docs/experiments/<name>.md".)

- [ ] **Step 1: Create the experiment config**

Create `configs/experiments/arch/pgpt.yaml`:

```yaml
# @package _global_
# pgpt — nGPT architecture with the explicit per-step weight projection REMOVED,
# co-trained with POET. Distinct from arch/ngpt_poet (which keeps vanilla nGPT's
# per-step renorm). See docs/superpowers/specs/2026-06-18-pgpt-design.md.
#
# Why no per-step all-weight renorm: POET parametrizes each trained linear as
# A·W_base·B (block-orthogonal A,B), preserving each matrix's singular spectrum
# exactly; the hidden state stays on the sphere via the runtime residual-blend /
# Q-K justnorms. The two sphere matrices POET does NOT wrap (token embedding +
# lm_head) keep a targeted per-step renorm via pgpt_optimizer_setup's hook.
#
# Distributed optimizer must be OFF (POET rejects it); train_pgpt_dev.sh passes
# parallelism.distributed_optimizer=false last (cluster config merges after this).
experiment:
  name: pgpt
  family: arch
  kind: pgpt
  description: |
    nGPT hypersphere architecture minus the per-step weight projection, trained
    with POET. 2D linears become POETLinear (frozen base + block-orthogonal delta
    oft_R). No per-step all-weight renorm; embedding+lm_head get a targeted renorm.
  references:
    - "Loshchilov et al. 2024 (arXiv:2410.01131)"
    - "POET"
  patches:
    - model_unfuse_linears    # unfuse fused qkv/fc1 at build time (pre-DDP)
    - poet_apply_to_model     # replace 2D linears with POETLinear (post-build)
    - poet_optimizer_setup    # route the POET optimizer (oft_R via Megatron-Adam)
    - poet_merge_step         # POET periodic merge (available; inert at merge_period=0)
    - pgpt_apply_spec         # pgpt layer spec + init normalize + post-step role map
    - pgpt_optimizer_setup    # no-WD scaling groups + embedding/lm_head renorm hook
    - training_log_eta        # prepend "ETA: HhMMm" to the per-iteration log
    - wandb_metric_normalize  # canonicalize W&B metric keys + add tokens_seen / step_time
  required_capabilities: []

optim:
  type: poet
  lr: 15.0e-4
  weight_decay: 0.0          # scaling params also zero-WD via pgpt_optimizer_setup
  betas: [0.9, 0.95]
  eps: 1.0e-8
  ngpt:                      # pgpt reuses the nGPT scaling-vector inits
    alpha_init: 0.05
    sqk_init: 1.0
    suv_init: 1.0
    sz_init: 1.0
    no_warmup: true
  poet:
    init_type: normalized
    merge_period: 0          # v1 default: no merge. Flip >0 to enable merges.
    scale: 0.5               # LR multiplier for the oft_R (POET linear) group

# Disable QK layernorm + dropout (conflict with the hypersphere forward); force MHA.
base:
  model:
    qk_norm: false
    attention_dropout: 0.0
    hidden_dropout: 0.0
    num_query_groups: ${base.model.num_attention_heads}
    transformer_impl: local
    unfuse_qkv: true
    unfuse_fc1: true
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1
```

- [ ] **Step 2: Create the dev script**

```bash
cd /lustre/fast/fast/zqiu/slm-research
cp scripts/train_ngpt_dev_poet.sh scripts/train_pgpt_dev.sh
sed -i \
  -e 's#experiment=arch/ngpt_poet#experiment=arch/pgpt#g' \
  -e 's/train_ngpt_dev_poet/train_pgpt_dev/g' \
  scripts/train_pgpt_dev.sh
```

Then open `scripts/train_pgpt_dev.sh` and confirm: (a) the `experiment=arch/pgpt` override is present, (b) `parallelism.distributed_optimizer=false` is still passed last, and (c) the comment header references pgpt. Fix the header comment wording if it still says nGPT.

- [ ] **Step 3: Verify the experiment resolves and emits both flags**

```bash
SLM_DRYRUN_PRINT=1 bash scripts/train_pgpt_dev.sh 2>/dev/null | tr ' ' '\n' | grep -E -- '--ngpt$|--poet$|experiment=arch/pgpt|distributed_optimizer'
```
Expected: lines containing `experiment=arch/pgpt`, `--ngpt`, `--poet`, and `distributed_optimizer=false`. (If the script has no `SLM_DRYRUN_PRINT` path, use `python -m launchers.train_megatron experiment=arch/pgpt base/family=llama3 base/scale=60m cluster=h100_de --dry-run` and grep the printed `command`.)

- [ ] **Step 4: Create the experiment doc (required by pre-commit)**

Create `docs/experiments/pgpt.md`:

```markdown
# pgpt — nGPT architecture minus the per-step weight projection, trained with POET

pgpt is the nGPT hypersphere architecture with the **explicit per-step weight
projection removed from the model**, co-trained with POET. It is a distinct base
model — NOT `arch/ngpt_poet`, which keeps vanilla nGPT (and its per-step renorm)
and merely swaps the optimizer.

See the design spec:
[docs/superpowers/specs/2026-06-18-pgpt-design.md](../superpowers/specs/2026-06-18-pgpt-design.md).

## Why drop the per-step renorm
POET parametrizes each trained linear as `A·W_base·B` with block-orthogonal
`A,B`, so it preserves each matrix's singular-value spectrum exactly — the
conditioning role nGPT's per-step projection played. Hidden states stay on the
sphere via the runtime residual-blend and Q/K `justnorm`s (activation ops POET
never touches). The two sphere matrices POET does not wrap (token embedding +
lm_head) keep a targeted per-step renorm installed by `pgpt_optimizer_setup`.

## Mechanism
- `pgpt_apply_spec` swaps the layer spec, stamps the nGPT config fields, runs the
  one-shot init normalize, and registers both the full role map and the
  embedding/lm_head post-step subset.
- `pgpt_optimizer_setup` cooperatively wraps `setup_model_and_optimizer`
  (`targets=()`): zero-WD for the scaling params, and a `optimizer.step` hook that
  re-projects only embedding + lm_head.
- `poet_merge_step` is included (no `train_step` collision, unlike `ngpt_poet`);
  inert at `merge_period=0`, flip `optim.poet.merge_period>0` to enable merges.

> **EXPERIMENTAL — GPU-smoke before trusting the loss.** Confirm the
> `[pgpt] optimizer setup …` log line and the `--ngpt`/`--poet` arg evidence both
> appear, and that POET `ortho_err` stays bounded.

## Run
`scripts/train_pgpt_dev.sh` (60m llama3 backbone, single GPU; passes
`parallelism.distributed_optimizer=false`).
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/arch/pgpt.yaml scripts/train_pgpt_dev.sh docs/experiments/pgpt.md
git commit -m "feat(pgpt): add experiment config, dev script, experiment doc"
```

---

## Task 8: Full CPU test sweep + GPU hand-off

**Files:** none (verification + handoff only).

- [ ] **Step 1: Run the full pgpt CPU test suite**

```bash
$PY -m pytest tests/unit/test_pgpt_layer_block_forward.py \
  tests/unit/test_pgpt_layer_spec.py \
  tests/unit/test_pgpt_apply_spec.py \
  tests/unit/test_pgpt_optimizer_groups.py \
  tests/unit/test_pgpt_renorm_hook.py \
  tests/unit/test_pgpt_patch_registry.py \
  tests/unit/test_pgpt_megatron_args.py -v
```
Expected: all passed.

- [ ] **Step 2: Confirm no nGPT coupling anywhere in the pgpt surface**

```bash
! grep -rn 'src\.model\.ngpt\|src\.specs\.ngpt_layer_spec\|src\.patches\.ngpt_' \
    src/model/pgpt src/specs/pgpt_layer_spec.py src/patches/pgpt_apply_spec.py \
    src/patches/pgpt_optimizer_setup.py && echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 3: Hand the GPU smoke command to the user**

Report (do NOT run — GPU is the user's to launch):

```
GPU smoke (single 60m run, ~a few minutes on one H100):
  codexlog pgpt_smoke bash scripts/train_pgpt_dev.sh training.train_iters=20

Confirm in the log:
  - "[pgpt] optimizer setup: zero-WD scaling groups + embedding/lm_head renorm hook"
  - "[nGPT] applied spec + attached sz + registered weight-norm roles"  (pgpt reuses this log)
  - POET wrap log ("[POET] replaced N linears ...") and bounded ortho_err
  - loss decreases and no NaN at step 2 (the historical nGPT/POET OOM/NaN points)
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §2.5 distinct-from-ngpt_poet → Tasks 5/7 (separate patch set + config, no ngpt_poet reuse). §3 decision 1 (drop per-step, keep init) → Task 3 keeps `_normalize_now`, no normalize_step patch ported. Decision 2 (POET-required) → Task 3 Step 2. Decision 3 (targeted renorm via optimizer hook) → Task 4. Decision 4 (full fork) → Tasks 1–2 + constraint checks (Task 1 Step 5, Task 8 Step 2). Decision 5 (merge regime) → Task 7 patch list + `merge_period: 0`. §4.4 hook → Task 4. §4.5 enforcement → Task 3. §5 patch list + omissions → Task 7 config. §5.1 flag wiring → Task 6. §6 validation → Tasks 1,4,5,8.
- **Placeholder scan:** none — all code blocks are complete; forks use exact `cp`+`sed`.
- **Type consistency:** `classify_pgpt_scaling_params` (Task 4) used in Tasks 4 tests; `_pgpt_post_step_norm_role_map` produced in Task 3, consumed in Task 4; `build_pgpt_layer_spec` produced Task 2, consumed Task 3; `_pgpt_arch_args` produced/consumed Task 6. Class names `PGPTBlock/PGPTTransformerLayer/PGPTMLP` consistent across Tasks 1–3.
