# DeepSeek-3Bv2 (MQA + sandwich-norm) first-party port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Huawei DeepSeek-3Bv2 architecture (MQA + sandwich-norm + MoE) trainable through the first-party `launchers.train_megatron` on Megatron 0.17, with no dependence on `poet_torch_huawei/`.

**Architecture:** A new `deepseek_v3_mqa` family + `deepseek_3bv2` scale capture the architecture as config; sandwich-norm is added as an ngpt-shaped runtime patch (`sandwich_norm_apply`) that registers args, stamps the `TransformerConfig`, and swaps in a `SandwichTransformerLayer` subclass via the GPT layer spec. The subclass injects the post-norm through PyTorch **forward-hooks** on the attention/MLP submodules (post-norm the sub-layer output before the residual add), so no Megatron `forward` is copied. POET layers on top unchanged; the stability/monitor suite is a separate sub-project.

**Tech Stack:** Megatron-core 0.17, Hydra configs, slm-research patch registry, pytest. Runtime patches only — no edits to `third_party/Megatron-LM/` or `poet_torch_huawei/`.

**Spec:** [docs/superpowers/specs/2026-05-29-deepseek-v3-mqa-sandwich-port-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-29-deepseek-v3-mqa-sandwich-port-design.md)

**IMPORTANT — test execution:** Per project convention **the user runs tests/compute** (no working env here). After each test+impl step, give the exact `pytest` command and **wait for the user's pass/fail** rather than asserting success. The CPU unit tests (Tasks 1, 2, 4, 6, 7) are the authoritative pre-GPU coverage; the layer instantiation + end-to-end behavior is the GPU smoke (Task 9).

**Deferred-by-design (functional-equivalent scope):** the perf/numerics-only Huawei flags `--no-rope-fusion`, `--manual-gc[-interval]`, `--cross-entropy-fusion-impl native`, and `--make-vocab-size-divisible-by 3232` are intentionally NOT ported (they don't change training dynamics functionally). If the GPU smoke shows loss divergence from the Huawei run, revisit them. The stability/monitor suite is Sub-project 2.

---

## File structure

- **Modify** [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) — rotary-percent config-driven; emit sandwich + 2 MoE flags; decouple MTP from MLA.
- **Modify** [launchers/pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py) — register `--use-sandwich-norm` / `--attn-post-norm-scale` / `--ffn-post-norm-scale` in `add_slm_args`.
- **Create** [src/model/sandwich_norm_ops.py](/lustre/fast/fast/zqiu/slm-research/src/model/sandwich_norm_ops.py) — pure, CPU-testable hook + scale helpers.
- **Create** [src/model/sandwich_layer.py](/lustre/fast/fast/zqiu/slm-research/src/model/sandwich_layer.py) — `SandwichTransformerLayer` subclass.
- **Create** [src/patches/sandwich_norm_apply.py](/lustre/fast/fast/zqiu/slm-research/src/patches/sandwich_norm_apply.py) — args + config stamp + layer-spec swap.
- **Create** [configs/base/family/deepseek_v3_mqa.yaml](/lustre/fast/fast/zqiu/slm-research/configs/base/family/deepseek_v3_mqa.yaml), [configs/base/scale/deepseek_3bv2.yaml](/lustre/fast/fast/zqiu/slm-research/configs/base/scale/deepseek_3bv2.yaml).
- **Modify** [configs/experiments/optim/adam.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/adam.yaml), [poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml), [muon_hybrid.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/muon_hybrid.yaml) — add `sandwich_norm_apply` to `patches:`.
- **Create** [scripts/train_deepseek.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_deepseek.sh).
- **Create** tests: `tests/unit/test_sandwich_norm.py`, `tests/unit/test_deepseek_v3_mqa_scale.py`; **modify** `tests/unit/test_megatron_args.py`, `tests/unit/test_patches_registry.py` (or a new patch test).

---

## Task 1: rotary-percent config-driven

**Files:**
- Modify: [src/utils/megatron_args.py:63](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L63)
- Test: [tests/unit/test_megatron_args.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_megatron_args.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_rotary_percent_is_config_driven():
    from src.utils.megatron_args import _model_args

    cfg = OmegaConf.create(
        {"base": {"model": _MIN_MODEL | {"rotary_percent": 0.25}}}
    )
    args = _model_args(cfg)
    assert args[args.index("--rotary-percent") + 1] == "0.25"


def test_rotary_percent_defaults_to_one():
    from src.utils.megatron_args import _model_args

    cfg = OmegaConf.create({"base": {"model": _MIN_MODEL}})
    args = _model_args(cfg)
    assert args[args.index("--rotary-percent") + 1] == "1.0"
```

Add this minimal-model helper near the top of the file (after imports) if not already present:

```python
_MIN_MODEL = {
    "num_layers": 2, "hidden_size": 64, "ffn_hidden_size": 128,
    "num_attention_heads": 4, "num_query_groups": 4, "head_dim": 16,
    "seq_length": 128, "normalization": "RMSNorm", "norm_epsilon": 1e-6,
    "positional_encoding": "rope", "rotary_base": 10000,
    "attention_dropout": 0.0, "hidden_dropout": 0.0, "init_method_std": 0.02,
    "tie_embeddings": True, "activation": "SwiGLU",
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_megatron_args.py::test_rotary_percent_is_config_driven -v`
Expected: FAIL — value is `"1.0"` (hardcoded), not `"0.25"`.

- [ ] **Step 3: Make rotary-percent config-driven**

In `src/utils/megatron_args.py`, change line 63 from:

```python
    _add(args, "--rotary-percent", 1.0)
```

to:

```python
    _add(args, "--rotary-percent", model.get("rotary_percent", 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_megatron_args.py -k rotary -v`
Expected: PASS (both). Also run the full file to confirm no regression: `python -m pytest tests/unit/test_megatron_args.py -v`. **Wait for the user.**

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(args): make --rotary-percent config-driven (base.model.rotary_percent)"
```

---

## Task 2: pure sandwich-norm ops

**Files:**
- Create: `src/model/sandwich_norm_ops.py`
- Test: `tests/unit/test_sandwich_norm.py`

These pure helpers are the CPU-testable core of the post-norm behavior.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sandwich_norm.py`:

```python
"""Tests for sandwich-norm post-norm ops (CPU)."""

import torch
import torch.nn as nn

from src.model.sandwich_norm_ops import apply_post_norm_scale, make_post_norm_hook


def test_post_norm_hook_norms_primary_output_preserves_rest():
    norm = nn.Linear(4, 4, bias=False)  # stand-in "norm"
    hook = make_post_norm_hook(norm)
    x = torch.ones(2, 4)
    new = hook(None, None, (x, None))
    assert torch.allclose(new[0], norm(x))
    assert new[1] is None


def test_post_norm_hook_handles_bare_tensor_output():
    norm = nn.Linear(4, 4, bias=False)
    hook = make_post_norm_hook(norm)
    x = torch.ones(2, 4)
    new = hook(None, None, x)
    assert torch.allclose(new, norm(x))


def test_apply_post_norm_scale_multiplies_weight():
    m = nn.LayerNorm(4)  # weight initialised to ones
    apply_post_norm_scale(m, 0.03)
    assert torch.allclose(m.weight, torch.full((4,), 0.03))


def test_apply_post_norm_scale_noop_at_one():
    m = nn.LayerNorm(4)
    apply_post_norm_scale(m, 1.0)
    assert torch.allclose(m.weight, torch.ones(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_sandwich_norm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.sandwich_norm_ops'`.

- [ ] **Step 3: Create the ops**

Create `src/model/sandwich_norm_ops.py`:

```python
"""Pure helpers for sandwich-norm (no Megatron import — CPU-safe).

Sandwich-norm applies a normalization to a sub-layer's *output* before the
residual add, with the norm weight scaled small at init. We inject it via a
forward-hook on the attention / MLP submodule so no Megatron forward is copied.
"""

from __future__ import annotations

import torch


def make_post_norm_hook(norm):
    """Build a forward-hook that post-norms a submodule's primary output.

    Megatron's attention / MLP modules return ``(output, bias)``; we normalize
    ``output`` and pass ``bias`` (and any further elements) through unchanged. A
    bare-tensor output is also supported (for tests / future modules).
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (norm(output[0]),) + tuple(output[1:])
        return norm(output)

    return hook


def apply_post_norm_scale(norm_module, scale: float) -> None:
    """Multiply a norm module's ``weight`` by ``scale`` in-place (no-op at 1.0).

    Matches the Huawei init: post-norm weights start small (e.g. 0.03) so the
    post-norm contributes near-identity at the start of training.
    """
    if scale != 1.0:
        with torch.no_grad():
            norm_module.weight.mul_(scale)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_sandwich_norm.py -v`
Expected: PASS (4 tests). **Wait for the user.**

- [ ] **Step 5: Commit**

```bash
git add src/model/sandwich_norm_ops.py tests/unit/test_sandwich_norm.py
git commit -m "feat(sandwich): pure post-norm hook + scale ops"
```

---

## Task 3: SandwichTransformerLayer subclass

**Files:**
- Create: `src/model/sandwich_layer.py`
- Test: `tests/unit/test_sandwich_norm.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_sandwich_norm.py`:

```python
def test_sandwich_layer_subclasses_transformer_layer():
    import pytest

    tl = pytest.importorskip("megatron.core.transformer.transformer_layer")
    from src.model.sandwich_layer import SandwichTransformerLayer

    assert issubclass(SandwichTransformerLayer, tl.TransformerLayer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_sandwich_norm.py::test_sandwich_layer_subclasses_transformer_layer -v`
Expected: FAIL (module missing) or SKIP if Megatron isn't importable on this box.

- [ ] **Step 3: Create the layer**

Create `src/model/sandwich_layer.py`:

```python
"""Sandwich-norm transformer layer for first-party Megatron 0.17.

Subclasses Megatron's ``TransformerLayer`` and, when ``config.use_sandwich_norm``
is set, adds a post-norm to the attention and MLP outputs *before* the residual
add (Huawei DeepSeek-3Bv2 "sandwich" norm). The post-norm is injected via a
forward-hook on ``self.self_attention`` / ``self.mlp`` so the (long, version-
coupled) ``_forward_attention`` / ``_forward_mlp`` methods are not copied.

No-op when ``use_sandwich_norm`` is false, so this class is safe as the default
GPT layer module.
"""

from __future__ import annotations

from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.sandwich_norm_ops import apply_post_norm_scale, make_post_norm_hook


def _sandwich_norm_cls():
    """TENorm if Transformer Engine is available, else WrappedTorchNorm.

    Mirrors the Huawei ``_get_sandwich_norm_impl``: TENorm imports even without
    TE (its __new__ raises at instantiation), so guard on HAVE_TE.
    """
    try:
        from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm

        if HAVE_TE:
            return TENorm
    except ImportError:
        pass
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    return WrappedTorchNorm


class SandwichTransformerLayer(TransformerLayer):
    """TransformerLayer + optional post-attention / post-MLP sandwich norm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        if not getattr(cfg, "use_sandwich_norm", False):
            return
        norm_cls = _sandwich_norm_cls()
        self.post_self_attn_layernorm = norm_cls(
            config=cfg, hidden_size=cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        self.post_mlp_layernorm = norm_cls(
            config=cfg, hidden_size=cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        apply_post_norm_scale(
            self.post_self_attn_layernorm, getattr(cfg, "attn_post_norm_scale", 1.0)
        )
        apply_post_norm_scale(self.post_mlp_layernorm, getattr(cfg, "ffn_post_norm_scale", 1.0))
        # Post-norm the sub-layer output before the bias-dropout-residual add.
        self.self_attention.register_forward_hook(
            make_post_norm_hook(self.post_self_attn_layernorm)
        )
        self.mlp.register_forward_hook(make_post_norm_hook(self.post_mlp_layernorm))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_sandwich_norm.py -v`
Expected: PASS (or the subclass test SKIPs if Megatron isn't importable here; it runs on the GPU box). **Wait for the user.**

- [ ] **Step 5: Commit**

```bash
git add src/model/sandwich_layer.py tests/unit/test_sandwich_norm.py
git commit -m "feat(sandwich): SandwichTransformerLayer with forward-hook post-norms"
```

---

## Task 4: megatron_args — emit sandwich + MoE-fusion flags; decouple MTP from MLA

**Files:**
- Modify: [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) (`_model_args`: sandwich after `--disable-bias-linear` ~L77; 2 MoE flags ~L131; MTP pulled out of the MLA block ~L100-105)
- Test: [tests/unit/test_megatron_args.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_megatron_args.py)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_sandwich_norm_flags_emitted_when_enabled():
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "use_sandwich_norm": True,
        "attn_post_norm_scale": 0.03,
        "ffn_post_norm_scale": 0.03,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--use-sandwich-norm" in args
    assert args[args.index("--attn-post-norm-scale") + 1] == "0.03"
    assert args[args.index("--ffn-post-norm-scale") + 1] == "0.03"


def test_sandwich_norm_flags_omitted_by_default():
    from src.utils.megatron_args import _model_args

    args = _model_args(OmegaConf.create({"base": {"model": _MIN_MODEL}}))
    assert "--use-sandwich-norm" not in args
    assert "--attn-post-norm-scale" not in args


def test_moe_router_fusion_and_layer_recompute_emitted():
    from src.utils.megatron_args import _model_args

    moe = {
        "enabled": True, "num_experts": 8, "layer_freq": "([1]*2)",
        "ffn_hidden_size": 128, "shared_expert_intermediate_size": 128,
        "router_load_balancing_type": "seq_aux_loss", "router_topk": 2,
        "token_dispatcher_type": "alltoall", "grouped_gemm": False,
        "aux_loss_coeff": 1e-4, "router_topk_scaling_factor": 2.5,
        "router_score_function": "sigmoid", "router_enable_expert_bias": True,
        "router_bias_update_rate": 1e-3, "router_dtype": "fp32",
        "permute_fusion": True, "router_fusion": True, "layer_recompute": True,
    }
    model = _MIN_MODEL | {"moe": moe}
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--moe-router-fusion" in args
    assert "--moe-layer-recompute" in args


def test_mtp_emitted_without_mla():
    # Review fix: MTP must emit for MQA (no MLA), where Huawei DeepSeek-3Bv2 uses it.
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "multi_latent_attention": False,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.3,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert args[args.index("--mtp-num-layers") + 1] == "1"
    assert args[args.index("--mtp-loss-scaling-factor") + 1] == "0.3"
    assert "--enable-experimental" in args


def test_mtp_still_emitted_with_mla():
    # Regression: the existing MLA path still emits MTP + experimental.
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "multi_latent_attention": True,
        "q_lora_rank": 64, "kv_lora_rank": 32, "qk_head_dim": 16,
        "qk_pos_emb_head_dim": 8, "v_head_dim": 16, "rotary_scaling_factor": 40,
        "mscale": 1.0, "mscale_all_dim": 1.0,
        "mtp_num_layers": 1, "mtp_loss_scaling_factor": 0.1,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--mtp-num-layers" in args
    assert "--enable-experimental" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_megatron_args.py::test_sandwich_norm_flags_emitted_when_enabled -v`
Expected: FAIL — `--use-sandwich-norm` not found.

- [ ] **Step 3: Emit the sandwich flags**

In `_model_args`, immediately after `_add(args, "--disable-bias-linear")` (~L77), add:

```python
    if bool(model.get("use_sandwich_norm", False)):
        args.append("--use-sandwich-norm")
        _add(args, "--attn-post-norm-scale", model.get("attn_post_norm_scale", 1.0))
        _add(args, "--ffn-post-norm-scale", model.get("ffn_post_norm_scale", 1.0))
```

- [ ] **Step 4: Emit the two missing MoE flags**

In the MoE block, after `_maybe_bool(args, "--moe-permute-fusion", moe.permute_fusion)` (~L131), add:

```python
        _maybe_bool(args, "--moe-router-fusion", moe.get("router_fusion", False))
        _maybe_bool(args, "--moe-layer-recompute", moe.get("layer_recompute", False))
```

- [ ] **Step 5: Decouple MTP emission from MLA**

Currently `--mtp-num-layers` / `--mtp-loss-scaling-factor` / `--enable-experimental` are emitted
**only inside** the `if multi_latent_attention:` block, so the MQA family would never get MTP.
In `_model_args`: (a) delete these two lines from the MLA `for key, flag in (...)` tuple:

```python
            ("mtp_num_layers", "--mtp-num-layers"),
            ("mtp_loss_scaling_factor", "--mtp-loss-scaling-factor"),
```

(b) delete the `_add(args, "--enable-experimental")` line at the end of the MLA block; (c) add a
standalone block immediately after the MLA `if` block:

```python
    # MTP is independent of MLA (Huawei DeepSeek-3Bv2 uses MTP with MQA).
    if model.get("mtp_num_layers", None) is not None:
        _add(args, "--mtp-num-layers", model.mtp_num_layers)
        _add(args, "--mtp-loss-scaling-factor", model.get("mtp_loss_scaling_factor", 0.1))
    if (
        bool(model.get("multi_latent_attention", False))
        or model.get("mtp_num_layers", None) is not None
    ):
        _add(args, "--enable-experimental")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_megatron_args.py -k "sandwich or moe_router_fusion or mtp" -v`
Expected: PASS (incl. the two MTP tests). Then run the full file to confirm no regression to the
existing MLA `deepseek_v3` emission. **Wait for the user.**

- [ ] **Step 7: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(args): emit sandwich flags + moe router-fusion/recompute; decouple MTP from MLA"
```

---

## Task 5: register sandwich CLI args + sandwich_norm_apply patch

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (`add_slm_args`)
- Create: `src/patches/sandwich_norm_apply.py`
- Test: `tests/unit/test_pretrain_gpt_slm.py`, `tests/unit/test_patch_sandwich_norm.py`

**Review fix:** slm-custom CLI flags are registered in `add_slm_args` (where `--poet`,
`--unfuse-qkv`, `--ngpt` live) — NOT by wrapping a Megatron-internal arg adder. So the three
sandwich flags go there. **Review fix:** the MoE decoder is built via
`get_gpt_decoder_block_spec` (gpt_builders.py:57), **not** `_get_transformer_layer_spec`, so the
spec swap must cover the block spec's `.layer_specs` (and the MTP spec) — otherwise sandwich-norm
silently never applies to this MoE model. The patch stamps config + swaps the layer class across
all spec paths (modeled on `ngpt_apply_spec`).

- [ ] **Step 1: Write the failing arg test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_sandwich_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--use-sandwich-norm",
         "--attn-post-norm-scale", "0.03", "--ffn-post-norm-scale", "0.03"]
    )
    assert args.use_sandwich_norm is True
    assert args.attn_post_norm_scale == 0.03
    assert args.ffn_post_norm_scale == 0.03
```

- [ ] **Step 2: Register the args in add_slm_args**

In `launchers/pretrain_gpt_slm.py` `add_slm_args`, after the `--unfuse-fc1` line, add:

```python
    # Sandwich-norm (architectural; applied by the sandwich_norm_apply patch).
    group.add_argument("--use-sandwich-norm", action="store_true")
    group.add_argument("--attn-post-norm-scale", type=float, default=1.0)
    group.add_argument("--ffn-post-norm-scale", type=float, default=1.0)
```

- [ ] **Step 3: Run the arg test**

Run: `python -m pytest tests/unit/test_pretrain_gpt_slm.py -v`
Expected: PASS (all). **Wait for the user.**

- [ ] **Step 4: Write the failing patch-registration test**

Create `tests/unit/test_patch_sandwich_norm.py`:

```python
"""Tests for the sandwich_norm_apply patch registration."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.sandwich_norm_apply", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.sandwich_norm_apply", None)


def test_patch_registers_on_arguments_and_gpt_builder():
    importlib.import_module("src.patches.sandwich_norm_apply")
    reg = registered_patches()
    assert "sandwich_norm_apply" in reg
    targets = reg["sandwich_norm_apply"].targets
    assert any("gpt_builder" in t for t in targets)
    assert any("core_transformer_config_from_args" in t for t in targets)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_patch_sandwich_norm.py -v`
Expected: FAIL — module missing.

- [ ] **Step 6: Create the patch**

Create `src/patches/sandwich_norm_apply.py`:

```python
"""Patch: stamp sandwich-norm config + swap in SandwichTransformerLayer.

Mirrors src/patches/ngpt_apply_spec.py. The CLI args (--use-sandwich-norm,
--attn-post-norm-scale, --ffn-post-norm-scale) are registered in add_slm_args;
this patch only (1) stamps them onto the TransformerConfig and (2) swaps the
transformer-layer class to SandwichTransformerLayer across every spec path used
by gpt_builder — the dense spec, the MoE decoder block spec (.layer_specs), and
the MTP layer spec. All gated on args.use_sandwich_norm (no-op otherwise).
Megatron imports happen inside apply() so importing this module is CPU-safe.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = (
    "gpt_builders.gpt_builder",
    "megatron.training.arguments.core_transformer_config_from_args",
)
logger = logging.getLogger(__name__)


@register_patch(name="sandwich_norm_apply", targets=_TARGET)
def apply() -> None:
    # ---- stamp config from args ----
    from megatron.training import arguments as _ma

    _orig_cfg = _ma.core_transformer_config_from_args

    def _wrapped_cfg(args, *a, **kw):
        config = _orig_cfg(args, *a, **kw)
        if getattr(args, "use_sandwich_norm", False):
            config.use_sandwich_norm = True
            config.attn_post_norm_scale = float(getattr(args, "attn_post_norm_scale", 1.0))
            config.ffn_post_norm_scale = float(getattr(args, "ffn_post_norm_scale", 1.0))
        return config

    _ma.core_transformer_config_from_args = _wrapped_cfg

    # ---- swap the layer class across all spec paths ----
    import gpt_builders as _gb
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from src.model.sandwich_layer import SandwichTransformerLayer

    def _sandwichify(spec):
        """Set spec.module = SandwichTransformerLayer wherever a base
        TransformerLayer spec appears (single ModuleSpec, a list of them, or a
        TransformerBlockSubmodules with .layer_specs)."""
        if spec is None:
            return spec
        if isinstance(spec, (list, tuple)):
            for s in spec:
                _sandwichify(s)
        elif hasattr(spec, "layer_specs"):
            _sandwichify(spec.layer_specs)
        elif getattr(spec, "module", None) is TransformerLayer:
            spec.module = SandwichTransformerLayer
        return spec

    _orig_builder = _gb.gpt_builder
    # Names of the spec-producing functions gpt_builder calls (dense / MoE / MTP).
    _spec_fns = ("get_gpt_decoder_block_spec", "_get_transformer_layer_spec",
                 "get_gpt_decoder_layer_specs")

    def _wrapped_builder(args, *a, **kw):
        if not getattr(args, "use_sandwich_norm", False):
            return _orig_builder(args, *a, **kw)
        originals = {}
        for name in _spec_fns:
            fn = getattr(_gb, name, None)
            if fn is None:
                continue
            originals[name] = fn

            def _make(orig):
                def wrapped(*aa, **kk):
                    return _sandwichify(orig(*aa, **kk))
                return wrapped

            setattr(_gb, name, _make(fn))
        try:
            model = _orig_builder(args, *a, **kw)
        finally:
            for name, fn in originals.items():
                setattr(_gb, name, fn)
        logger.info("[sandwich] swapped layer class on all spec paths (dense/MoE/MTP)")
        return model

    _gb.gpt_builder = _wrapped_builder
```

> **Implementer note:** verify the three spec-fn names exist in
> `third_party/Megatron-LM/gpt_builders.py` (they are imported/defined there in 0.17:
> `get_gpt_decoder_block_spec`, `_get_transformer_layer_spec`, `get_gpt_decoder_layer_specs`).
> The Task 9 smoke must show `[sandwich] swapped layer class ...` AND a post-norm parameter
> (e.g. `post_self_attn_layernorm.weight`) present on every decoder layer.

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_patch_sandwich_norm.py -v`
Expected: PASS (registration only; `apply()` is not called on CPU). **Wait for the user.**

- [ ] **Step 8: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/patches/sandwich_norm_apply.py \
  tests/unit/test_pretrain_gpt_slm.py tests/unit/test_patch_sandwich_norm.py
git commit -m "feat(sandwich): CLI args + sandwich_norm_apply patch (config stamp + MoE/MTP-aware layer swap)"
```

---

## Task 6: new family + scale configs

**Files:**
- Create: `configs/base/family/deepseek_v3_mqa.yaml`, `configs/base/scale/deepseek_3bv2.yaml`
- Test: `tests/unit/test_deepseek_v3_mqa_scale.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_deepseek_v3_mqa_scale.py`:

```python
"""Composition + arg-emission tests for the deepseek_v3_mqa family / deepseek_3bv2 scale."""

from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args


def _cfg():
    return _parse_overrides(
        ["base/family=deepseek_v3_mqa", "base/scale=deepseek_3bv2", "experiment=optim/adam"]
    )


def test_scale_resolves_mqa_and_sandwich():
    m = _cfg().base.model
    assert m.num_layers == 12
    assert m.hidden_size == 1280
    assert m.ffn_hidden_size == 7168
    assert m.num_attention_heads == 16
    assert m.head_dim == 384
    assert m.num_query_groups == 1
    assert m.multi_latent_attention is False
    assert m.use_sandwich_norm is True
    assert m.rotary_percent == 0.25
    assert m.moe.ffn_hidden_size == 896
    assert m.moe.router_topk == 6


def test_megatron_args_emit_mqa_sandwich_moe():
    args = build_megatron_args(_cfg())
    assert "--group-query-attention" in args
    assert args[args.index("--num-query-groups") + 1] == "1"
    assert args[args.index("--kv-channels") + 1] == "384"
    assert args[args.index("--rotary-percent") + 1] == "0.25"
    assert "--use-sandwich-norm" in args
    assert args[args.index("--attn-post-norm-scale") + 1] == "0.03"
    assert args[args.index("--moe-router-topk") + 1] == "6"
    assert args[args.index("--moe-ffn-hidden-size") + 1] == "896"
    assert "--multi-latent-attention" not in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_deepseek_v3_mqa_scale.py -v`
Expected: FAIL — family/scale configs not found (Hydra composition error).

- [ ] **Step 3: Create the family config**

Create `configs/base/family/deepseek_v3_mqa.yaml`:

```yaml
# @package _global_
# DeepSeek-3Bv2 sandwich-norm + MQA, ported from the Huawei stack
# (poet_torch_huawei/training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml).
# MQA (num_query_groups=1) + sandwich-norm — distinct from the MLA deepseek_v3 family.
base:
  family: deepseek_v3_mqa
  family_version: "3bv2_sandwich_mqa"
  reference: "Huawei DeepSeek-3Bv2 sandwich-mqa (de-vendored to first-party Megatron 0.17)"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-6
    activation: "SwiGLU"
    positional_encoding: "rope"
    rotary_base: 10000
    rotary_percent: 0.25
    qk_norm: true
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.006
    attention_backend: "flash"
    multi_latent_attention: false
    num_query_groups: 1
    tie_embeddings: false
    # sandwich norm (applied by the sandwich_norm_apply patch)
    use_sandwich_norm: true
    attn_post_norm_scale: 0.03
    ffn_post_norm_scale: 0.03
    # MTP
    mtp_num_layers: 1
    mtp_loss_scaling_factor: 0.3
    moe:
      enabled: true
      num_experts: 64
      layer_freq: "([0]*1+[1]*11)"
      ffn_hidden_size: 896
      shared_expert_intermediate_size: 1792
      router_load_balancing_type: "seq_aux_loss"
      router_topk: 6
      token_dispatcher_type: "alltoall"
      enable_deepep: false
      router_pre_softmax: false
      grouped_gemm: false
      aux_loss_coeff: 1.0e-4
      router_group_topk: null
      router_num_groups: null
      router_topk_scaling_factor: 2.5
      router_score_function: "sigmoid"
      router_enable_expert_bias: true
      router_bias_update_rate: 1.0e-3
      router_dtype: "fp32"
      permute_fusion: true
      router_fusion: true
      layer_recompute: true
  tokenizer:
    nominal_name: "deepseek-v3"
    nominal_vocab_size: 129280
```

- [ ] **Step 4: Create the scale config**

Create `configs/base/scale/deepseek_3bv2.yaml`:

```yaml
# @package _global_
# DeepSeek-3Bv2: 12L, hidden 1280, dense ffn 7168, 16 heads, head_dim 384 (MQA),
# MoE ffn 896 / shared 1792, seq 4096. Network size only; arch in family.
base:
  scale: "deepseek_3bv2"
  non_embedding_params: 3_000_000_000
  model:
    num_layers: 12
    hidden_size: 1280
    ffn_hidden_size: 7168
    num_attention_heads: 16
    head_dim: 384
    seq_length: 4096
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_deepseek_v3_mqa_scale.py -v`
Expected: PASS. **Wait for the user.**

- [ ] **Step 6: Commit**

```bash
git add configs/base/family/deepseek_v3_mqa.yaml configs/base/scale/deepseek_3bv2.yaml tests/unit/test_deepseek_v3_mqa_scale.py
git commit -m "feat(config): deepseek_v3_mqa family + deepseek_3bv2 scale (MQA + sandwich-norm)"
```

---

## Task 7: wire sandwich_norm_apply into experiment patch lists

**Files:**
- Modify: `configs/experiments/optim/adam.yaml`, `poet.yaml`, `muon_hybrid.yaml`
- Test: `tests/unit/test_deepseek_v3_mqa_scale.py` (append)

The patch is a no-op unless `use_sandwich_norm` is set, so adding it to these experiments is safe (same pattern as `model_unfuse_linears`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_deepseek_v3_mqa_scale.py`:

```python
def test_sandwich_patch_listed_in_experiments():
    for exp in ("optim/adam", "optim/poet", "optim/muon_hybrid"):
        cfg = _parse_overrides([f"experiment={exp}"])
        patches = list(cfg.experiment.patches)
        assert "sandwich_norm_apply" in patches, exp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_deepseek_v3_mqa_scale.py::test_sandwich_patch_listed_in_experiments -v`
Expected: FAIL — patch not in lists.

- [ ] **Step 3: Add the patch to each experiment**

In `configs/experiments/optim/adam.yaml`, `poet.yaml`, and `muon_hybrid.yaml`, add `- sandwich_norm_apply` to the `patches:` list (before `training_log_eta`). Example for `adam.yaml`:

```yaml
  patches:
    - model_unfuse_linears
    - sandwich_norm_apply    # no-op unless base.model.use_sandwich_norm
    - training_log_eta
```

(For `poet.yaml`, insert after `model_unfuse_linears`; for `muon_hybrid.yaml`, after `model_unfuse_linears`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_deepseek_v3_mqa_scale.py -v`
Expected: PASS. **Wait for the user.**

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/adam.yaml configs/experiments/optim/poet.yaml configs/experiments/optim/muon_hybrid.yaml tests/unit/test_deepseek_v3_mqa_scale.py
git commit -m "feat(experiments): list sandwich_norm_apply patch (no-op unless enabled)"
```

---

## Task 8: train_deepseek.sh launch script

**Files:**
- Create: `scripts/train_deepseek.sh`

- [ ] **Step 1: Create the script**

Create `scripts/train_deepseek.sh` (mirroring `scripts/train_adam.sh`; the WSD/LR/batch hyperparameters come from the Huawei full launcher):

```bash
#!/usr/bin/env bash
set -euo pipefail

# First-party DeepSeek-3Bv2 (MQA + sandwich-norm) training, de-vendored from
# poet_torch_huawei. Defaults to plain AdamW; pass experiment=optim/poet to add POET.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

python -m launchers.train_megatron \
  "base/family=deepseek_v3_mqa" \
  "base/scale=deepseek_3bv2" \
  "cluster=h100_de" \
  "experiment=optim/adam" \
  "scheduler=wsd" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=4" \
  "optim.lr=8.6e-4" \
  "optim.min_lr=7e-6" \
  "$@"
```

- [ ] **Step 2: Smoke-parse the config (dry, on the GPU box / user)**

Run: `bash scripts/train_deepseek.sh --help 2>&1 | head` is not meaningful; instead verify Hydra composition resolves:
`python -m launchers.train_megatron base/family=deepseek_v3_mqa base/scale=deepseek_3bv2 experiment=optim/adam scheduler=wsd --cfg job 2>&1 | tail -20` (if `--cfg job` is supported) — otherwise rely on Task 6 tests + Task 9 smoke. **User runs.**

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/train_deepseek.sh
git add scripts/train_deepseek.sh
git commit -m "feat(scripts): train_deepseek.sh — first-party DeepSeek-3Bv2 MQA launcher"
```

---

## Task 9: full suite, lint, GPU smoke (user-run)

**Files:** none (verification only).

- [ ] **Step 1: Run the unit suite**

Run:
```
python -m pytest tests/unit/test_sandwich_norm.py tests/unit/test_patch_sandwich_norm.py \
  tests/unit/test_deepseek_v3_mqa_scale.py tests/unit/test_megatron_args.py -v
```
Expected: all PASS (the Megatron-importing subclass test may SKIP on CPU). **Wait for the user.**

- [ ] **Step 2: Lint**

Run: `pre-commit run --files src/model/sandwich_norm_ops.py src/model/sandwich_layer.py src/patches/sandwich_norm_apply.py src/utils/megatron_args.py configs/base/family/deepseek_v3_mqa.yaml configs/base/scale/deepseek_3bv2.yaml scripts/train_deepseek.sh`
Expected: PASS (fix ruff issues, re-run). **Wait for the user.**

- [ ] **Step 3: Single-GPU smoke (user, GPU box)**

```bash
CUDA_VISIBLE_DEVICES=0 codexlog deepseek_mqa_smoke bash scripts/train_deepseek.sh \
  cluster=dev \
  training.global_batch_size=8 training.micro_batch_size=1 \
  base.model.seq_length=512 training.train_iters=10 training.log_interval=1
```
Expected: model builds (validates head_dim=384, MoE SequentialMLP, MTP, sandwich layer instantiation + forward-hook), `[sandwich] applied SandwichTransformerLayer spec` in the log, and ~10 steps run with finite decreasing loss and 0 nan/skipped. If `--use-sandwich-norm` errors as "unrecognized argument", fix the arg-group adder name in `sandwich_norm_apply._register_sandwich_cli_args` (Task 5 note). If OOM on one GPU, reduce experts via `base.model.moe.num_experts=8` for the smoke. **User runs and reports.**

- [ ] **Step 4: Arg reconciliation check (user, GPU box or full env)**

Dump the emitted argv and eyeball against the Huawei `MODEL_ARGS` for any *architecturally-meaningful* gap (the deferred perf flags in the header are expected to be absent):
```bash
python -m launchers.train_megatron base/family=deepseek_v3_mqa base/scale=deepseek_3bv2 \
  experiment=optim/adam scheduler=wsd --print-args 2>/dev/null || true
```
(If no `--print-args`, read the launched `torchrun ... pretrain_gpt_slm` command from the smoke log.) Confirm MQA, sandwich, MoE, MTP, rotary-percent 0.25 are present. **User runs.**

- [ ] **Step 5: Update CHANGELOG + experiment doc**

Add a CHANGELOG entry and (optional) a `docs/experiments/` note describing the new family/scale and the sandwich-norm patch.

```bash
git add CHANGELOG.md docs/
git commit -m "docs(deepseek): changelog + notes for first-party DeepSeek-3Bv2 MQA port"
```

---

## Self-review notes

- **Spec coverage:** §4 rotary (T1) + sandwich emission (T4) + MoE reconcile (T4); §5a family (T6); §5b scale (T6); §5c sandwich patch (T2 ops, T3 layer, T5 patch); §5d args (T1,T4); §5e recipe/launch (T8); §5f tests (T2,T3,T5,T6,T9); §6 files (all); §7 risks (T9 smoke gates head_dim, sandwich forward, SequentialMLP, MTP, reconciliation). Covered.
- **Type/name consistency:** `make_post_norm_hook`, `apply_post_norm_scale`, `SandwichTransformerLayer`, `_sandwich_norm_cls`, patch name `sandwich_norm_apply`, config fields `use_sandwich_norm`/`attn_post_norm_scale`/`ffn_post_norm_scale`, family `deepseek_v3_mqa`, scale `deepseek_3bv2` — used identically across tasks.
- **Implementer note (Task 5):** the arg-group adder name `_add_network_size_args` is the expected 0.17 name; verify against `third_party/Megatron-LM/megatron/training/arguments.py` and wrap whichever `_add_*_args` `parse_args` invokes if it differs. The Task 9 smoke catches an "unrecognized argument" mismatch.
