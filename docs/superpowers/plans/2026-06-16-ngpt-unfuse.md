# nGPT Unfuse Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make nGPT train correctly with `unfuse_qkv`/`unfuse_fc1` enabled, so its weight-matrix decomposition (separate q/k/v and gate/up-style projections) matches the unfused baselines (adam, muon) for a like-for-like architecture bake-off.

**Architecture:** Three independent fixes. (A) Repair a patch-binding bug that silently disables the entire nGPT spec whenever `model_unfuse_linears` is present. (B) Extend the nGPT weight-normalization role map to recognize the unfused parameter names. (C) Teach nGPT's custom MLP to build *split* `u`/`v` projections natively (the generic unfuse cannot handle it because of the learned `suv` scaling and nGPT's reversed `[u|v]` packing). nGPT attention already unfuses correctly through the existing `model_unfuse_linears` patch — only the role map needs the new names.

**Tech Stack:** PyTorch, Megatron-LM (TP=1), pytest. CPU-runnable unit tests for all logic; Megatron-dependent numerics parity + the 60m GPU smoke are handed to the operator (Megatron import requires the CUDA env).

---

## Background / Root Cause (read before starting)

The newest nGPT run ([runs/ngpt-llama3-60m-s42-20260616T094854Z](/lustre/fast/fast/zqiu/slm-research/runs/ngpt-llama3-60m-s42-20260616T094854Z)) trained a **plain llama3 baseline, not nGPT**: the checkpoint has zero nGPT params (`sz`/`sqk`/`suv`/`alpha`) and the `[nGPT] applied spec` log line is absent. Three facts drive this plan:

1. **Patch-binding bug (the dominant failure).** `apply_patches` sorts patch names alphabetically ([_registry.py:94](/lustre/fast/fast/zqiu/slm-research/src/patches/_registry.py#L94)). `model_unfuse_linears` (**m**) therefore applies before `ngpt_apply_spec` (**n**), and `model_unfuse_linears.apply()` eagerly does `import pretrain_gpt` ([model_unfuse_linears.py:29](/lustre/fast/fast/zqiu/slm-research/src/patches/model_unfuse_linears.py#L29)). That import pulls in `gpt_builders`, freezing **two** by-value bindings to the *originals* before `ngpt_apply_spec` wraps them:
   - `pretrain_gpt.gpt_builder` (via `from gpt_builders import gpt_builder`, [pretrain_gpt.py:24](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/pretrain_gpt.py#L24)) — the launcher passes this stale original (`partial(mg.model_provider, mg.gpt_builder)`, [pretrain_gpt_slm.py:254](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L254)), so the nGPT-wrapped **builder** never runs (no spec swap).
   - `gpt_builders.core_transformer_config_from_args` (via `from megatron.training.arguments import ...`, [gpt_builders.py:20](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/gpt_builders.py#L20), called bare at [:34](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/gpt_builders.py#L34) since `model_provider` passes `config=None`, [model_provider.py:63](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/model_provider.py#L63)) — so the nGPT **config stamp** (`softmax_scale=sqrt(head_dim)` and, for this plan, `config.unfuse_fc1`) never lands.

   Both are wrapped on a *different* module than their caller, so a `from X import Y` copy made before the wrap stays stale; the normalize/optimizer patches are unaffected because their wrapped attr and caller live in the same module (`megatron.training.training`). **This breaks nGPT even when fused**, and the config half must also be fixed or Task 4's MLP unfuse won't activate. (Task 1)

2. **Role map only knows fused names.** `_NORM_ROLES_BY_SUFFIX` ([ngpt_apply_spec.py:115](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py#L115)) matches only `linear_qkv`/`linear_fc1` (plus `linear_proj`/`linear_fc2`). Under unfuse the qkv splits into `linear_q`/`linear_k`/`linear_v`, so those rows would never be normalized — and `_register_ngpt_norm_roles`' lower-bound assertion would still pass (it only checks a minimum), silently skipping normalization. (Task 2)

3. **nGPT's MLP cannot use the generic unfuse.** `NGPTMLP` packs `[u|v]` in one `nn.Linear` and applies a learned `suv` across all `2·ffn` columns *before* chunking ([mlp.py:42-62](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py#L42)). The generic unfuse (a) only recognizes `ColumnParallelLinear`/`TEColumnParallelLinear`, not `nn.Linear`, so it skips the MLP; and (b) its replacement forward ([unfuse_linears.py:180](/lustre/fast/fast/zqiu/slm-research/src/model/unfuse_linears.py#L180)) drops `suv` and assumes Megatron's `[gate|up]` ordering (silu on the *first* half), whereas nGPT packs `[u|v]` (silu on the *second* half, `u·silu(v)`). So nGPT builds the split MLP **natively**. (Tasks 3-4)

**Key invariant that makes this safe:** nGPT's weight-norm is row-wise and `ngpt_adamw` is per-parameter, so fused and unfused nGPT are the *same model* given identical weights. Every parity test below asserts exactly that: copy fused weights into the split modules, assert identical forward output.

**Why attention needs no code change:** nGPT forces MHA (`num_query_groups == num_attention_heads`, [ngpt.yaml:52](/lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml#L52)). The existing `_unfused_qkv_forward` ([unfuse_linears.py:233-260](/lustre/fast/fast/zqiu/slm-research/src/model/unfuse_linears.py#L233)) reshapes q/k to per-head and calls `self.q_layernorm`/`self.k_layernorm`, which are nGPT's `QKHyperNorm` (sqk) — so sqk is preserved. Only the role map (Task 2) must learn the `linear_q/k/v` names.

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py) | Add `_rebind_if_stale` (rebinds both stale wraps), extend `_NORM_ROLES_BY_SUFFIX`, stamp `config.unfuse_fc1` | 1, 2, 4 |
| [src/model/ngpt/mlp.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py) | `NGPTMLPBody`/`NGPTMLP` gain a native unfused (split `u`/`v`) construction + forward | 3, 4 |
| [tests/unit/test_ngpt_patch_binding.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_patch_binding.py) | NEW — unit test for the stale-binding rebind helper | 1 |
| [tests/unit/test_ngpt_role_map.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_role_map.py) | NEW — unit test for role-map suffix matching (fused + unfused) | 2, 5 |
| [tests/unit/test_ngpt_mlp.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_mlp.py) | Extend — fused-vs-unfused MLP forward parity | 3 |
| [configs/experiments/arch/ngpt.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml) | Already has unfuse on; update the stale "untested" comment | 6 |

No new module files — keeping nGPT-specific logic in the existing nGPT package and the generic unfuse module untouched (it stays generic).

---

### Task 1: Fix the patch-binding bug (both wraps)

`ngpt_apply_spec` wraps two functions on *other* modules: `gpt_builders.gpt_builder` (read by the launcher as `pretrain_gpt.gpt_builder`) and `megatron.training.arguments.core_transformer_config_from_args` (read by `gpt_builders` as a bare by-value name). When `model_unfuse_linears` imports `pretrain_gpt`→`gpt_builders` first, both consumers freeze by-value copies of the *originals*. A single generic helper rebinds any already-imported module that still holds the original. Extracted as a pure helper so it's CPU-testable without importing Megatron.

**Files:**
- Modify: [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py) (add helper + two calls inside `apply()`)
- Test: [tests/unit/test_ngpt_patch_binding.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_patch_binding.py) (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ngpt_patch_binding.py`:

```python
"""Unit test for rebinding stale by-value imports of functions ngpt_apply_spec
wraps (the model_unfuse_linears-before-ngpt ordering bug). Covers both the
gpt_builder (in pretrain_gpt) and core_transformer_config_from_args (in
gpt_builders) bindings via the one generic helper."""

import sys
import types

from src.patches.ngpt_apply_spec import _rebind_if_stale


def _orig():  # sentinel originals
    return "orig"


def _wrapped():
    return "wrapped"


def test_rebinds_module_holding_original():
    fake = types.ModuleType("pretrain_gpt")
    fake.gpt_builder = _orig
    sys.modules["pretrain_gpt"] = fake
    try:
        _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
        assert sys.modules["pretrain_gpt"].gpt_builder is _wrapped
    finally:
        del sys.modules["pretrain_gpt"]


def test_rebinds_config_function_in_gpt_builders():
    fake = types.ModuleType("gpt_builders")
    fake.core_transformer_config_from_args = _orig
    sys.modules["gpt_builders"] = fake
    try:
        _rebind_if_stale("gpt_builders", "core_transformer_config_from_args", _orig, _wrapped)
        assert sys.modules["gpt_builders"].core_transformer_config_from_args is _wrapped
    finally:
        del sys.modules["gpt_builders"]


def test_noop_when_module_not_imported():
    sys.modules.pop("pretrain_gpt", None)
    # Must not raise and must not create the module.
    _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
    assert "pretrain_gpt" not in sys.modules


def test_does_not_rebind_a_foreign_object():
    other = lambda: "other"  # noqa: E731
    fake = types.ModuleType("pretrain_gpt")
    fake.gpt_builder = other
    sys.modules["pretrain_gpt"] = fake
    try:
        _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
        # Only rebinds if it currently holds the captured original.
        assert sys.modules["pretrain_gpt"].gpt_builder is other
    finally:
        del sys.modules["pretrain_gpt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ngpt_patch_binding.py -q`
Expected: FAIL — `ImportError: cannot import name '_rebind_if_stale'`.

- [ ] **Step 3: Add the helper and call it for both wraps**

In [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py), add a module-level helper after the imports (near the top, before `apply`):

```python
def _rebind_if_stale(module_name: str, attr: str, orig, wrapped) -> None:
    """Rebind a stale by-value import of a function we just wrapped.

    ``ngpt_apply_spec`` wraps functions on one module that consumers captured
    by value (``from X import Y``) on another: ``gpt_builder`` (consumed as
    ``pretrain_gpt.gpt_builder`` by the launcher) and
    ``core_transformer_config_from_args`` (consumed bare inside
    ``gpt_builders``). If a patch sorting before us (``model_unfuse_linears``)
    imported those modules first, the copies are frozen to the originals.
    Rebind ``module_name.attr`` to ``wrapped`` — but only if it still holds the
    ``orig`` we wrapped, so we never clobber a different wrapper. No-op (and no
    import) if the module is not yet loaded; in that ordering the consumer
    imports after us and binds the wrapped function naturally.
    """
    import sys

    m = sys.modules.get(module_name)
    if m is not None and getattr(m, attr, None) is orig:
        setattr(m, attr, wrapped)
```

Then add the two calls inside `apply()`:

- Immediately after `_ma.core_transformer_config_from_args = _wrapped_cfg`:

```python
    _rebind_if_stale(
        "gpt_builders", "core_transformer_config_from_args", _orig_cfg, _wrapped_cfg
    )
```

- Immediately after `_gb.gpt_builder = _wrapped_builder`:

```python
    _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig_builder, _wrapped_builder)
```

(`_orig_cfg`/`_wrapped_cfg` and `_orig_builder`/`_wrapped_builder` are already in scope at those points.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_ngpt_patch_binding.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/ngpt_apply_spec.py tests/unit/test_ngpt_patch_binding.py
git commit -m "$(cat <<'EOF'
fix(ngpt): rebind stale gpt_builder + config wraps when unfuse pre-imports them
EOF
)"
```

---

### Task 2: Extend the weight-norm role map to unfused names

Add the unfused suffixes so nGPT normalizes the split q/k/v and u/v rows. Row-normalizing the split matrices is bit-identical to row-normalizing the fused ones (same rows), so this is backward-compatible: fused models match the old suffixes, unfused models match the new ones.

**Files:**
- Modify: [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py:115-128) (`_NORM_ROLES_BY_SUFFIX`)
- Test: [tests/unit/test_ngpt_role_map.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_role_map.py) (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ngpt_role_map.py`:

```python
"""Role-map suffix matching for nGPT weight normalization (fused + unfused)."""

from src.patches.ngpt_apply_spec import _match_role


def test_fused_names_map_to_roles():
    assert _match_role("decoder.layers.0.self_attention.linear_qkv.weight") == "rows"
    assert _match_role("decoder.layers.0.self_attention.linear_proj.weight") == "cols"
    assert _match_role("decoder.layers.0.mlp.linear_fc1.weight") == "rows"
    assert _match_role("decoder.layers.0.mlp.linear_fc2.weight") == "cols"


def test_unfused_qkv_names_map_to_rows():
    base = "decoder.layers.3.self_attention."
    assert _match_role(base + "linear_q.weight") == "rows"
    assert _match_role(base + "linear_k.weight") == "rows"
    assert _match_role(base + "linear_v.weight") == "rows"


def test_unfused_mlp_uv_names_map_to_rows():
    base = "decoder.layers.3.mlp."
    assert _match_role(base + "linear_fc1_u.weight") == "rows"
    assert _match_role(base + "linear_fc1_v.weight") == "rows"


def test_layer_norm_weight_and_unrelated_params_do_not_match():
    # TE LayerNorm-fused weight must not be mistaken for a matrix to normalize.
    assert _match_role("decoder.layers.0.self_attention.linear_qkv.layer_norm_weight") is None
    assert _match_role("decoder.final_layernorm.weight") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ngpt_role_map.py -q`
Expected: FAIL on `test_unfused_qkv_names_map_to_rows` / `test_unfused_mlp_uv_names_map_to_rows` (returns `None`).

- [ ] **Step 3: Add the unfused suffixes**

In [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py:115), change `_NORM_ROLES_BY_SUFFIX` from:

```python
_NORM_ROLES_BY_SUFFIX: dict[tuple[str, ...], str] = {
    # Embedding row = per-token vector -> unit norm along hidden.
    ("embedding", "word_embeddings", "weight"): "rows",
    # LM head row = per-vocab vector -> unit norm along hidden.
    ("output_layer", "weight"): "rows",
    # Q/K/V projection rows = per-output-channel vectors.
    ("linear_qkv", "weight"): "rows",
    # Attention output projection columns = per-input-channel vectors.
    ("linear_proj", "weight"): "cols",
    # SwiGLU c_fc rows = per-output-channel vectors (gate+up concat).
    ("linear_fc1", "weight"): "rows",
    # SwiGLU mlp_c_proj columns = per-input-channel vectors.
    ("linear_fc2", "weight"): "cols",
}
```

to:

```python
_NORM_ROLES_BY_SUFFIX: dict[tuple[str, ...], str] = {
    # Embedding row = per-token vector -> unit norm along hidden.
    ("embedding", "word_embeddings", "weight"): "rows",
    # LM head row = per-vocab vector -> unit norm along hidden.
    ("output_layer", "weight"): "rows",
    # Q/K/V projection rows = per-output-channel vectors (fused).
    ("linear_qkv", "weight"): "rows",
    # ...and the unfused split (model_unfuse_linears splits linear_qkv).
    ("linear_q", "weight"): "rows",
    ("linear_k", "weight"): "rows",
    ("linear_v", "weight"): "rows",
    # Attention output projection columns = per-input-channel vectors.
    ("linear_proj", "weight"): "cols",
    # SwiGLU c_fc rows = per-output-channel vectors (fused gate+up concat).
    ("linear_fc1", "weight"): "rows",
    # ...and nGPT's native unfused split (NGPTMLP builds u/v separately).
    ("linear_fc1_u", "weight"): "rows",
    ("linear_fc1_v", "weight"): "rows",
    # SwiGLU mlp_c_proj columns = per-input-channel vectors.
    ("linear_fc2", "weight"): "cols",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_ngpt_role_map.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/ngpt_apply_spec.py tests/unit/test_ngpt_role_map.py
git commit -m "$(cat <<'EOF'
feat(ngpt): recognize unfused q/k/v and u/v matrices in weight-norm role map
EOF
)"
```

---

### Task 3: Native unfused construction + forward in NGPTMLP

`NGPTMLPBody` gains an `unfuse` flag. When set, it builds `linear_fc1_u`/`linear_fc1_v` (each `[ffn, hidden]`) instead of the packed `linear_fc1` `[2*ffn, hidden]`, and keeps `suv` whole (shape `2*ffn`, so the param is byte-identical to the fused case and the optimizer's no-decay grouping is unchanged). The forward slices `suv` into its `[:ffn]`/`[ffn:]` halves. Parity is exact: given `linear_fc1_u.weight == fused.weight[:ffn]` and `linear_fc1_v.weight == fused.weight[ffn:]`, output is identical.

**Files:**
- Modify: [src/model/ngpt/mlp.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py)
- Test: [tests/unit/test_ngpt_mlp.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_mlp.py) (extend)

- [ ] **Step 1: Write the failing parity test**

Append to `tests/unit/test_ngpt_mlp.py`:

```python
def test_ngpt_mlp_body_unfused_matches_fused():
    """Splitting linear_fc1 into u/v (and slicing suv) is bit-identical to the
    packed forward, given the same weights. This is the fairness invariant:
    fused nGPT == unfused nGPT."""
    torch.manual_seed(1)
    n_embd = 16
    n_inner = 4 * n_embd
    base_scale = 1.0 / (n_embd**0.5)

    fused = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
        unfuse=False,
    )
    unfused = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
        unfuse=True,
    )
    # Copy fused weights into the split projections + shared suv/fc2.
    unfused.linear_fc1_u.weight.data.copy_(fused.linear_fc1.weight.data[:n_inner])
    unfused.linear_fc1_v.weight.data.copy_(fused.linear_fc1.weight.data[n_inner:])
    unfused.linear_fc2.weight.data.copy_(fused.linear_fc2.weight.data)
    unfused.suv.param.data.copy_(fused.suv.param.data)

    x = torch.randn(2, 5, n_embd)
    assert torch.allclose(fused(x), unfused(x), atol=1e-6)


def test_ngpt_mlp_body_unfused_param_count_matches_fused():
    n_embd, n_inner = 16, 64
    kw = dict(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=1.0 / (n_embd**0.5),
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
    )
    fused = NGPTMLPBody(unfuse=False, **kw)
    unfused = NGPTMLPBody(unfuse=True, **kw)
    assert sum(p.numel() for p in fused.parameters()) == sum(
        p.numel() for p in unfused.parameters()
    )
    # Split names exist; packed name does not.
    assert hasattr(unfused, "linear_fc1_u") and hasattr(unfused, "linear_fc1_v")
    assert not hasattr(unfused, "linear_fc1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ngpt_mlp.py -q`
Expected: FAIL — `NGPTMLPBody.__init__() got an unexpected keyword argument 'unfuse'`.

- [ ] **Step 3: Implement the unfused path in NGPTMLPBody**

In [src/model/ngpt/mlp.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py), replace the `NGPTMLPBody` class body (the `__init__` and `forward`, lines 25-62) with:

```python
class NGPTMLPBody(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        base_scale: float,
        suv_init_value: float,
        suv_init_scaling: float,
        dtype: torch.dtype = torch.bfloat16,
        unfuse: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self._n_embd_sqrt = float(self.hidden_size) ** 0.5
        self.unfuse = bool(unfuse)

        # nGPT reference packs c_fc with 2*ffn columns: [u_half | v_half].
        # Unfused: two separate [ffn, hidden] projections holding those halves.
        # suv stays a single (2*ffn,) vector either way (sliced in forward),
        # so the param is identical to the fused case and the optimizer's
        # no-decay grouping (keyed on the module name "suv") is unaffected.
        if self.unfuse:
            self.linear_fc1_u = nn.Linear(
                self.hidden_size, self.ffn_hidden_size, bias=False, dtype=dtype
            )
            self.linear_fc1_v = nn.Linear(
                self.hidden_size, self.ffn_hidden_size, bias=False, dtype=dtype
            )
            nn.init.normal_(self.linear_fc1_u.weight, mean=0.0, std=base_scale)
            nn.init.normal_(self.linear_fc1_v.weight, mean=0.0, std=base_scale)
        else:
            self.linear_fc1 = nn.Linear(
                self.hidden_size, 2 * self.ffn_hidden_size, bias=False, dtype=dtype
            )
            nn.init.normal_(self.linear_fc1.weight, mean=0.0, std=base_scale)

        self.linear_fc2 = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False, dtype=dtype)
        nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=base_scale)

        self.suv = LearnedScaling(
            shape=(2 * self.ffn_hidden_size,),
            init_value=suv_init_value,
            init_scaling=suv_init_scaling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference effective suv: param * (init_value/init_scaling) * sqrt(n_embd).
        suv = (self.suv.scaled_value() * self._n_embd_sqrt)
        ffn = self.ffn_hidden_size
        if self.unfuse:
            u = self.linear_fc1_u(x)
            v = self.linear_fc1_v(x)
            u = suv[:ffn].to(u.dtype) * u
            v = suv[ffn:].to(v.dtype) * v
        else:
            uv = self.linear_fc1(x)
            uv = suv.to(uv.dtype) * uv
            u, v = uv.chunk(2, dim=-1)
        return self.linear_fc2(u * functional.silu(v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_ngpt_mlp.py -q`
Expected: PASS (all tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/model/ngpt/mlp.py tests/unit/test_ngpt_mlp.py
git commit -m "$(cat <<'EOF'
feat(ngpt): native unfused u/v construction in NGPTMLPBody
EOF
)"
```

---

### Task 4: Wire `config.unfuse_fc1` so the real build selects the unfused MLP

The model build path must tell `NGPTMLP` to use the split projections. `--unfuse-fc1` lands on `args.unfuse_fc1`; stamp it onto the `TransformerConfig` (alongside the other nGPT fields) so `NGPTMLP.__init__` can read it. With the MLP self-splitting, `model_unfuse_linears` finds no `linear_fc1` on `NGPTMLP` and skips it (it still splits the attention `linear_qkv`), so there is no double-unfuse.

**Files:**
- Modify: [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py:44-60) (`_wrapped_cfg`)
- Modify: [src/model/ngpt/mlp.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py:79-91) (`NGPTMLP.__init__`)
- Test: [tests/unit/test_ngpt_mlp.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_mlp.py) (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_ngpt_mlp.py`:

```python
def test_ngpt_mlp_reads_unfuse_flag_from_config():
    """NGPTMLP (the Megatron-instantiable subclass) selects split projections
    when config.unfuse_fc1 is set."""
    from src.model.ngpt.mlp import NGPTMLP

    class _Cfg:
        hidden_size = 16
        ffn_hidden_size = 64
        ngpt_base_scale = 1.0 / (16**0.5)
        ngpt_suv_init = 1.0
        bf16 = False
        params_dtype = torch.float32

    fused = NGPTMLP(_Cfg())
    assert hasattr(fused, "linear_fc1") and not hasattr(fused, "linear_fc1_u")

    cfg_unfused = _Cfg()
    cfg_unfused.unfuse_fc1 = True
    unfused = NGPTMLP(cfg_unfused)
    assert hasattr(unfused, "linear_fc1_u") and hasattr(unfused, "linear_fc1_v")
    assert not hasattr(unfused, "linear_fc1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ngpt_mlp.py::test_ngpt_mlp_reads_unfuse_flag_from_config -q`
Expected: FAIL — `unfused` still has `linear_fc1` (flag ignored).

- [ ] **Step 3a: Read the flag in NGPTMLP**

In [src/model/ngpt/mlp.py](/lustre/fast/fast/zqiu/slm-research/src/model/ngpt/mlp.py:79), change `NGPTMLP.__init__` to pass the flag through. Replace its `super().__init__(...)` call with one that adds `unfuse=`:

```python
    def __init__(self, config, submodules=None, **kwargs) -> None:
        hidden = int(config.hidden_size)
        dtype = getattr(config, "params_dtype", None)
        if dtype is None:
            dtype = torch.bfloat16 if getattr(config, "bf16", True) else torch.float32
        super().__init__(
            hidden_size=hidden,
            ffn_hidden_size=int(config.ffn_hidden_size),
            base_scale=float(getattr(config, "ngpt_base_scale", 1.0 / (hidden**0.5))),
            suv_init_value=float(getattr(config, "ngpt_suv_init", 1.0)),
            suv_init_scaling=1.0,
            dtype=dtype,
            unfuse=bool(getattr(config, "unfuse_fc1", False)),
        )
```

- [ ] **Step 3b: Stamp the flag onto the config**

In [src/patches/ngpt_apply_spec.py](/lustre/fast/fast/zqiu/slm-research/src/patches/ngpt_apply_spec.py:59), inside `_wrapped_cfg`, just before `config.ngpt = True`, add:

```python
        # NGPTMLP reads this to build split u/v projections (parity with the
        # unfused baselines). Attention unfuse is handled by model_unfuse_linears.
        config.unfuse_fc1 = bool(getattr(args, "unfuse_fc1", False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ngpt_mlp.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/patches/ngpt_apply_spec.py src/model/ngpt/mlp.py tests/unit/test_ngpt_mlp.py
git commit -m "$(cat <<'EOF'
feat(ngpt): select unfused MLP from config.unfuse_fc1
EOF
)"
```

---

### Task 5: Role-map registration covers an unfused model

Confirm `_register_ngpt_norm_roles` matches every normalizable matrix in a *fully unfused* nGPT model (7 per layer: q, k, v, proj, fc1_u, fc1_v, fc2) plus the embedding, and that its lower-bound assertion holds. This guards against a future rename silently dropping a matrix from normalization.

**Files:**
- Test: [tests/unit/test_ngpt_role_map.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_role_map.py) (extend)

- [ ] **Step 1: Write the test**

Append to `tests/unit/test_ngpt_role_map.py`:

```python
import torch

from src.patches.ngpt_apply_spec import _register_ngpt_norm_roles


class _FakeUnfusedModel:
    """Plain stand-in: `_register_ngpt_norm_roles` only calls
    `named_parameters()` and assigns `_ngpt_norm_role_map`. Param names mirror a
    fully-unfused nGPT model; distinct tensors so they key the role dict
    uniquely."""

    def __init__(self, layers=2, hidden=8, ffn=16):
        self._params = []
        for i in range(layers):
            for sub, rows in [
                (f"decoder.layers.{i}.self_attention.linear_q", hidden),
                (f"decoder.layers.{i}.self_attention.linear_k", hidden),
                (f"decoder.layers.{i}.self_attention.linear_v", hidden),
                (f"decoder.layers.{i}.self_attention.linear_proj", hidden),
                (f"decoder.layers.{i}.mlp.linear_fc1_u", ffn),
                (f"decoder.layers.{i}.mlp.linear_fc1_v", ffn),
                (f"decoder.layers.{i}.mlp.linear_fc2", hidden),
            ]:
                self._params.append((sub + ".weight", torch.nn.Parameter(torch.randn(rows, hidden))))
        self._params.append(
            ("embedding.word_embeddings.weight", torch.nn.Parameter(torch.randn(10, hidden)))
        )

    def named_parameters(self, *a, **k):
        return list(self._params)


def test_register_roles_matches_all_unfused_matrices():
    model = _FakeUnfusedModel(layers=2, hidden=8, ffn=16)
    _register_ngpt_norm_roles(model, expected_layers=2)
    roles = model._ngpt_norm_role_map
    # 7 matrices/layer * 2 + 1 embedding = 15 normalizable params.
    assert len(roles) == 15
    n_rows = sum(1 for r in roles.values() if r == "rows")
    n_cols = sum(1 for r in roles.values() if r == "cols")
    # rows: q,k,v,fc1_u,fc1_v (5/layer) + embedding = 11 ; cols: proj,fc2 (2/layer) = 4
    assert (n_rows, n_cols) == (11, 4)
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/unit/test_ngpt_role_map.py -q`
Expected: PASS (depends only on Task 2's suffixes; should pass green).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_ngpt_role_map.py
git commit -m "$(cat <<'EOF'
test(ngpt): role-map registration covers a fully unfused model
EOF
)"
```

---

### Task 6: Config comment + full CPU suite + GPU smoke handoff

The config already enables unfuse ([ngpt.yaml:20,56-57](/lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml#L20)); the only change is refreshing the now-stale "untested" comment. Then run the full nGPT + unfuse + patch CPU suite, and hand the operator the Megatron-dependent numerics parity test and the 60m GPU smoke (Megatron import requires the CUDA env; the operator runs cluster jobs).

**Files:**
- Modify: [configs/experiments/arch/ngpt.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml:53-57)

- [ ] **Step 1: Update the stale comment**

In [configs/experiments/arch/ngpt.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml:53), replace:

```yaml
    # Unfuse fused qkv/fc1 into separate projections (applied by the
    # model_unfuse_linears patch above). NOTE: untested with nGPT's custom
    # layer spec + weight normalization — smoke before relying on it.
    unfuse_qkv: true
    unfuse_fc1: true
```

with:

```yaml
    # Unfuse into separate projections for parity with the adam/muon baselines.
    # qkv: split by the model_unfuse_linears patch (preserves sqk via q/k_layernorm).
    # fc1: NGPTMLP builds split u/v natively from config.unfuse_fc1 (the generic
    # unfuse can't handle nGPT's suv + [u|v] packing). Both are weight-norm
    # registered; fused and unfused nGPT are the same model (row-norm is
    # fusion-invariant). See docs/superpowers/plans/2026-06-16-ngpt-unfuse.md.
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 2: Run the full CPU test suite for the touched areas**

Run:
```bash
python -m pytest tests/unit/test_ngpt_mlp.py tests/unit/test_ngpt_role_map.py \
  tests/unit/test_ngpt_patch_binding.py tests/unit/test_ngpt_optimizer_groups.py \
  tests/unit/test_ngpt_attention.py tests/unit/test_ngpt_layer_spec.py \
  tests/unit/test_ngpt_output_scaling.py tests/unit/test_ngpt_step_parity.py \
  tests/unit/test_unfuse_linears.py -q
```
Expected: all PASS. (If `pytest` is missing: `python -m pip install pytest` per the env note.)

- [ ] **Step 3: Commit**

```bash
git add configs/experiments/arch/ngpt.yaml
git commit -m "$(cat <<'EOF'
docs(ngpt): note unfuse parity wiring in experiment config
EOF
)"
```

- [ ] **Step 4: Operator handoff — Megatron numerics parity (GPU/CUDA env)**

These require `source load_cuda13_2_nccl_env.sh` and a GPU; hand to the operator:

```bash
# 1) Full nGPT Megatron layer parity (extend if it does not yet cover unfuse):
python -m pytest tests/numerics/test_ngpt_megatron_layer_parity.py -q

# 2) 60m smoke — confirms the [nGPT] apply log now fires and the checkpoint
#    carries sz/sqk/suv/alpha + split q/k/v + fc1_u/fc1_v params:
bash scripts/train_ngpt.sh base/scale=60m   # operator's standard nGPT smoke
```
Acceptance: the training log shows `[nGPT] applied spec + attached sz + registered weight-norm roles` and `[unfuse] ... linear_qkv -> q/k/v` for all layers; the saved checkpoint param list contains `linear_q/linear_k/linear_v`, `mlp.linear_fc1_u/linear_fc1_v`, and the nGPT scaling params; val/loss tracks (or beats) the earlier fused-nGPT validation curve.

---

## Self-Review Notes

- **Spec coverage:** binding bug (both stale wraps — builder + config) → Task 1; role map → Tasks 2, 5; nGPT MLP unfuse → Tasks 3, 4; attention unfuse → no code (existing patch) + role map (Task 2); config comment + suite + GPU → Task 6.
- **Type/name consistency:** split modules are named `linear_fc1_u` / `linear_fc1_v` everywhere (role map, NGPTMLPBody, tests); `suv` stays a single `(2*ffn,)` `LearnedScaling` so `classify_ngpt_param_groups` (keyed on module name `"suv"`) is unchanged — no optimizer-grouping edit needed.
- **Dependency note:** Task 4's MLP unfuse only fires if Task 1's *config* rebind lands (`config.unfuse_fc1` is read off the stamped `TransformerConfig`). Task 1 also fixes `softmax_scale`, which has no safe default for nGPT. Do Task 1 first.
- **Coverage boundary (deliberate):** the SelfAttention + `QKHyperNorm` + unfuse integration and the config-stamp landing cannot be unit-tested on CPU (they need a real Megatron build, which requires the CUDA env). They are covered by Task 6 Step 4's Megatron numerics parity + the acceptance checks on the 60m smoke log/checkpoint.
- **`titan_init` interaction (benign):** `titan_init` re-inits by suffix and does not know `linear_fc1_u/v`, so it leaves them at `NGPTMLPBody`'s own `normal(0, base_scale)` init (it *does* re-init the fused `linear_fc1`). This is harmless for nGPT — the one-shot + per-step row normalization discards init magnitude, and only the (valid, randomly drawn) row directions survive. Fused and unfused nGPT therefore draw different MLP directions, exactly as two separate init code paths always would; both are valid nGPT inits. No `titan_init` change needed.
- **Open item to confirm during Task 6 Step 4:** whether `tests/numerics/test_ngpt_megatron_layer_parity.py` already parametrizes over unfuse; if not, the operator (or a follow-up) should add an unfused case mirroring Task 3's parity assertion at the full-layer level.
