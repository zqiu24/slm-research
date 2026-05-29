# POET (huawei vendored stack) — `poet_split_fc1` + divisibility hard-error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On branch `huawei`, give POET's SwiGLU MLP a true gate/up split (each branch its own frozen weight + `oft_R` + rotations) driven purely by `config.poet_split_fc1` inside the `MLP` class, harden the adapter's name-matcher against substring collisions, turn the silent divisibility skip into a hard error, and finally simplify the existing qkv split to the same config-only pattern.

**Architecture:** All edits are in the **vendored** `poet_torch_huawei/` stack (Megatron-core 0.14). The fc1 split is done in `MLP.__init__`/`MLP.forward` by reading `config.poet_split_fc1` and building two `ColumnParallelLinear`s from the *same* `submodules.linear_fc1` spec — so dense FFN, routed `SequentialMLP` experts, and the shared expert (all the `MLP` class) get it from one change, while GroupedMLP fused experts (not `MLP`) are untouched. No spec-builder threading. The matcher and divisibility changes live in `poet_adapter/adapter.py`. A final isolated task removes the now-redundant `poet_split_qkv` spec-level scaffolding so qkv matches fc1's config-only style.

**Tech Stack:** Megatron-core 0.14 (vendored), `poet_torch` (Cayley/Triton), the `poet` conda env (torch 2.8 / triton 3.4, no TE → `--transformer-impl local`), torchrun, pytest (for the two CPU-runnable adapter tests).

**Execution environment note.** Per memory `feedback_no_local_test_run`, there is no working training env in the harness — the **user runs every test/smoke on the `poet` node and reports back**. The two adapter unit tests (Tasks 6, 7) are pure-Python (CPU, no CUDA/Megatron build) and the user runs them with `pytest`; the model-level fc1 equivalence is only verifiable in the single-GPU smoke (Task 8). Plan steps that "run" something are dispatched to the user.

**Source of truth:** branch `huawei` @ current HEAD (`4db5412`). Spec: [docs/superpowers/specs/2026-05-29-poet-huawei-split-fc1-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-29-poet-huawei-split-fc1-design.md).

---

## File Structure

```
poet_torch_huawei/
├── megatron/core/transformer/transformer_config.py   # MODIFY — add poet_split_fc1 field (Task 1)
├── megatron/training/arguments.py                     # MODIFY — derive poet_split_fc1 from use_poet (Task 2)
├── megatron/core/transformer/mlp.py                   # MODIFY — __init__ split build (T3), forward split path (T4), sharded_state_dict + backward_dw guard (T5)
├── megatron/core/poet_adapter/adapter.py              # MODIFY — _name_matches dot-bound (T6), divisibility hard-error in _try_attach + _try_attach_te (T7)
├── megatron/core/models/gpt/gpt_layer_specs.py        # MODIFY — drop redundant qkv spec scaffolding (T9, final)
└── tests_poet/                                        # NEW dir (lives with the vendored stack, run under PYTHONPATH=poet_torch_huawei)
    └── test_adapter_unit.py                           # NEW — CPU unit tests for _name_matches (T6) + divisibility raise (T7)
```

**Responsibilities:**
- `transformer_config.py` / `arguments.py` — config + arg plumbing; `poet_split_fc1` auto-on with `--use-poet`, no user-facing argparse flag (mirrors `poet_split_qkv`).
- `mlp.py` — the entire fc1 split: construction, forward, checkpoint. Single locus covering dense/routed/shared.
- `adapter.py` — wrapping-selection robustness (matcher) and fail-loud divisibility.
- `gpt_layer_specs.py` — only touched in the final qkv-cleanup task.
- `tests_poet/test_adapter_unit.py` — the two genuinely CPU-runnable tests.

**Why the fc1 split does NOT need spec/experts edits:** under `--transformer-impl local`, the dense MLP, the shared expert ([moe_module_specs.py:52,62](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/gpt/moe_module_specs.py#L52)), and the routed `SequentialMLP` experts ([backends.py:110](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/backends.py#L110)) all hand `MLP` a `ColumnParallelLinear` as `submodules.linear_fc1`. `MLP.__init__` already receives `config`. So reading `config.poet_split_fc1` inside `MLP` and building two linears *from that same spec* covers all three with one change. GroupedMLP/TEGroupedMLP fused experts are not the `MLP` class (their `sharded_state_dict` at [experts.py:937](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/moe/experts.py#L937) is separate and never splits), and `--moe-grouped-gemm` is off for the target config.

---

## Task 1: Add the `poet_split_fc1` config field

**Files:**
- Modify: `poet_torch_huawei/megatron/core/transformer/transformer_config.py:163`

- [ ] **Step 1: Add the field next to `poet_split_qkv`**

In [transformer_config.py:163-164](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/transformer_config.py#L163-L164), the current code is:
```python
    poet_split_qkv: bool = False
    """Build self-attention Q, K and V projections as separate modules for POET/POET-X."""

    gated_linear_unit: bool = False
```
Insert the new field immediately after the `poet_split_qkv` docstring:
```python
    poet_split_qkv: bool = False
    """Build self-attention Q, K and V projections as separate modules for POET/POET-X."""

    poet_split_fc1: bool = False
    """Build the SwiGLU MLP gate and up projections as separate modules for POET/POET-X
    (so each branch gets its own frozen weight, oft_R, and rotations instead of one fused
    block-diagonal rotation entangling gate and up). Only meaningful with gated_linear_unit."""

    gated_linear_unit: bool = False
```

- [ ] **Step 2: Verify the field parses**

Run (user, on the `poet` node):
```bash
source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -c "from megatron.core.transformer.transformer_config import TransformerConfig as T; import dataclasses; assert any(f.name=='poet_split_fc1' for f in dataclasses.fields(T)); print('poet_split_fc1 field OK')"
```
Expected: `poet_split_fc1 field OK`.

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/transformer/transformer_config.py
git commit -m "feat(huawei): add poet_split_fc1 TransformerConfig field"
```

---

## Task 2: Auto-enable `poet_split_fc1` whenever `--use-poet` is set

**Files:**
- Modify: `poet_torch_huawei/megatron/training/arguments.py:1185`

- [ ] **Step 1: Derive the flag from `use_poet`**

In [arguments.py:1185](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/training/arguments.py#L1185), the current line is:
```python
    kw_args['poet_split_qkv'] = bool(getattr(args, 'use_poet', False))
```
Add the fc1 line immediately after it:
```python
    kw_args['poet_split_qkv'] = bool(getattr(args, 'use_poet', False))
    kw_args['poet_split_fc1'] = bool(getattr(args, 'use_poet', False))
```
(No argparse entry — `poet_split_qkv` has none either; both are derived purely from `--use-poet`.)

- [ ] **Step 2: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/training/arguments.py
git commit -m "feat(huawei): force poet_split_fc1 on with --use-poet"
```

---

## Task 3: Build separate gate/up linears in `MLP.__init__`

**Files:**
- Modify: `poet_torch_huawei/megatron/core/transformer/mlp.py:92-109`

- [ ] **Step 1: Replace the fused `linear_fc1` build with a split-aware build**

In [mlp.py:92-109](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L92-L109), the current code is:
```python
        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
        )
```
Replace it with (note `ffn_hidden_size` here is the per-branch width *before* the SwiGLU doubling; we capture it for the split path, then keep the original doubling for the fused path):
```python
        # Per-branch width (gate / up each get this). For a gated linear unit the
        # fused layer doubles the output (gate stacked on up); the split path
        # instead builds two separate linears of this width.
        # see https://arxiv.org/pdf/2002.05202.pdf
        branch_ffn_hidden_size = ffn_hidden_size

        # POET gate/up split: build linear_fc1_gate + linear_fc1_up as two
        # independent ColumnParallelLinears from the SAME submodule spec, so each
        # branch gets its own frozen weight + POET orbit (no fused block-diagonal
        # rotation entangling gate and up). Driven purely by config — no spec
        # threading needed because dense / routed-SequentialMLP / shared experts
        # all hand us a ColumnParallelLinear here.
        self.split_fc1 = bool(
            getattr(self.config, "poet_split_fc1", False)
            and self.config.gated_linear_unit
        )

        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        if self.split_fc1:
            self.linear_fc1_gate = build_module(
                submodules.linear_fc1,
                self.input_size,
                branch_ffn_hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                tp_comm_buffer_name="fc1",
                tp_group=tp_group,
            )
            self.linear_fc1_up = build_module(
                submodules.linear_fc1,
                self.input_size,
                branch_ffn_hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                tp_comm_buffer_name="fc1",
                tp_group=tp_group,
            )
        else:
            self.linear_fc1 = build_module(
                submodules.linear_fc1,
                self.input_size,
                ffn_hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=is_expert,
                tp_comm_buffer_name="fc1",
                tp_group=tp_group,
            )
```

- [ ] **Step 2: Verify the split branch constructs (deferred to the Task-8 smoke)**

`MLP` is not CPU-importable without a CUDA Megatron build (see the note in [tests/unit/test_poet_layers.py:1-6](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_layers.py#L1-L6)), so there is no standalone unit check here — the construction is exercised by the smoke in Task 8 (which prints the wrapped-layer inventory). Do a syntax check only:
```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
python -m py_compile megatron/core/transformer/mlp.py && echo "mlp.py compiles"
```
Expected: `mlp.py compiles`.

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/transformer/mlp.py
git commit -m "feat(huawei): build split gate/up fc1 linears in MLP.__init__ under poet_split_fc1"
```

---

## Task 4: Use the split linears in `MLP.forward`

**Files:**
- Modify: `poet_torch_huawei/megatron/core/transformer/mlp.py:127-194`

- [ ] **Step 1: Add the split forward path at the top of `forward`**

In [mlp.py:127-132](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L127-L132), the current code is:
```python
    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")
```
Replace it with a split short-circuit that computes `act(gate) * up` and jumps straight to fc2, leaving the entire fused path below untouched:
```python
    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        if getattr(self, "split_fc1", False):
            # POET gate/up split: two independent projections, then SwiGLU.
            # Each linear has its own frozen weight + POET orbit, so the gate
            # and up branches are never entangled by a shared rotation.
            nvtx_range_push(suffix="linear_fc1_gate_up")
            gate, _ = self.linear_fc1_gate(hidden_states)
            up, _ = self.linear_fc1_up(hidden_states)
            nvtx_range_pop(suffix="linear_fc1_gate_up")
            intermediate_parallel = self.activation_func(gate) * up
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            nvtx_range_push(suffix="linear_fc2")
            output, output_bias = self.linear_fc2(intermediate_parallel)
            nvtx_range_pop(suffix="linear_fc2")
            if per_token_scale is not None:
                assert output_bias is None, "Bias is not supported with per_token_scale"
            return output, output_bias

        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")
```
The fused activation/bias-fusion/fc2 block (lines below, [mlp.py:134-194](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L134-L194)) is unchanged and only runs when `split_fc1` is False. The split path matches the non-fused `glu()` semantics (`activation_func(gate) * up`, [mlp.py:172-176](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L172-L176)); forgoing the fused `bias_swiglu_impl` is fine since the target config runs `--disable-bias-linear: true` (no bias) and POET freezes the base weight regardless.

- [ ] **Step 2: Syntax check**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
python -m py_compile megatron/core/transformer/mlp.py && echo "mlp.py compiles"
```
Expected: `mlp.py compiles`.

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/transformer/mlp.py
git commit -m "feat(huawei): split gate/up SwiGLU forward path in MLP under poet_split_fc1"
```

---

## Task 5: Guard checkpoint sharding + `backward_dw` for the split path

**Files:**
- Modify: `poet_torch_huawei/megatron/core/transformer/mlp.py:202-215`

- [ ] **Step 1: `sharded_state_dict` — skip the swiglu factory for split halves**

The factory at [mlp.py:204-209](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L204-L209) splits the *fused* doubled weight for TP. With the split path the two halves are already separate plain `ColumnParallelLinear`s and must serialize verbatim — the factory must not run on them. The current loop is:
```python
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
```
Because `name` iterates `self._modules` and split mode registers `linear_fc1_gate`/`linear_fc1_up` (never `linear_fc1`), the `name == "linear_fc1"` guard is already False for split halves — so no factory is applied to them automatically. Add an explicit assertion-comment guard so intent is clear and a future rename can't reintroduce the bug:
```python
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            # Only the FUSED linear_fc1 needs the swiglu split factory. Split
            # gate/up halves (linear_fc1_gate / linear_fc1_up) are already
            # separate ColumnParallelLinears and serialize verbatim.
            if (
                self.config.gated_linear_unit
                and name == "linear_fc1"
                and not getattr(self, "split_fc1", False)
            ):
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
```

- [ ] **Step 2: `backward_dw` — handle the split linears**

The current method at [mlp.py:213-215](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L213-L215) is:
```python
    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()
```
With split mode there is no `self.linear_fc1`, so this would `AttributeError` for any caller that invokes it. Make it split-aware:
```python
    def backward_dw(self):
        self.linear_fc2.backward_dw()
        if getattr(self, "split_fc1", False):
            self.linear_fc1_gate.backward_dw()
            self.linear_fc1_up.backward_dw()
        else:
            self.linear_fc1.backward_dw()
```

- [ ] **Step 3: Syntax check**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
python -m py_compile megatron/core/transformer/mlp.py && echo "mlp.py compiles"
```
Expected: `mlp.py compiles`.

- [ ] **Step 4: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/transformer/mlp.py
git commit -m "fix(huawei): split-aware sharded_state_dict + backward_dw in MLP"
```

---

## Task 6: Harden `_name_matches` to dot-bounded token matching (TDD)

**Files:**
- Modify: `poet_torch_huawei/megatron/core/poet_adapter/adapter.py:500-504`
- Create: `poet_torch_huawei/tests_poet/test_adapter_unit.py`

This is genuinely CPU-runnable: `adapter.py`'s only module-level imports are `torch` / `torch.nn` ([adapter.py:49-50](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L49-L50)); all Megatron imports are deferred inside functions, so importing the module and calling `_name_matches` needs no CUDA build.

- [ ] **Step 1: Write the failing test**

Create `poet_torch_huawei/tests_poet/test_adapter_unit.py`:
```python
"""CPU unit tests for the POET adapter's pure-Python helpers.

Run from the vendored stack root so `megatron` resolves to the vendored copy:
    cd poet_torch_huawei && PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v
adapter.py only imports torch at module level (Megatron imports are deferred
inside functions), so these tests need no CUDA / Megatron build.
"""

import pytest

from megatron.core.poet_adapter.adapter import _name_matches

# The default POET leaf-exclusion list (adapter.install_poet_in_model).
EXCLUDE = ("lm_head", "output_layer", "embedding", "word_embeddings", "router", "gate", "mtp")


@pytest.mark.parametrize(
    "name",
    [
        "module.decoder.layers.1.mlp.router.weight",
        "module.mtp.layers.0.transformer_layer.mlp.router.weight",
        "decoder.layers.0.output_layer",
        "embedding.word_embeddings",
    ],
)
def test_excluded_names_still_match(name):
    assert _name_matches(name, EXCLUDE) is True


@pytest.mark.parametrize(
    "name",
    [
        # The new split halves must NOT be caught by the "gate" pattern.
        "decoder.layers.0.mlp.linear_fc1_gate",
        "decoder.layers.1.mlp.experts.local_experts.3.linear_fc1_gate",
        "decoder.layers.0.mlp.linear_fc1_up",
        "decoder.layers.1.self_attention.linear_q",
    ],
)
def test_fc1_split_halves_not_excluded(name):
    assert _name_matches(name, EXCLUDE) is False


def test_ancestor_dotted_pattern_still_matches():
    # exclude_ancestors carries the dotted ".experts." pattern; it must keep
    # matching real expert paths while not matching shared_experts.
    assert _name_matches("decoder.layers.1.mlp.experts.local_experts.0.linear_fc1", (".experts.",)) is True
    assert _name_matches("decoder.layers.1.mlp.shared_experts.linear_fc1", (".experts.",)) is False
```

- [ ] **Step 2: Run the test — verify it FAILS on the current matcher**

Run (user, on the `poet` node):
```bash
source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v
```
Expected: `test_fc1_split_halves_not_excluded[...linear_fc1_gate]` FAILS — the current raw-substring matcher returns True for `"gate" in "linear_fc1_gate"`. (The dotted-ancestor case may also fail.)

- [ ] **Step 3: Implement the dot-bounded matcher**

In [adapter.py:500-504](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L500-L504), the current code is:
```python
def _name_matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    lower = name.lower()
    return any(pat.lower() in lower for pat in patterns)
```
Replace with dot-bounded token matching (a pattern matches a whole dot-delimited segment, or a dotted substring like `.experts.`, never an arbitrary substring):
```python
def _name_matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    # Dot-bound the haystack so a bare leaf token (e.g. "gate") matches a path
    # SEGMENT (".gate.") rather than any substring ("linear_fc1_gate"). Patterns
    # that already carry dots (e.g. ".experts.") keep their intended meaning.
    # Prevents the substring-collision class of bug where a new layer name
    # containing an exclusion token gets silently dropped from POET wrapping.
    hay = "." + name.lower() + "."
    return any(("." + pat.lower().strip(".") + ".") in hay for pat in patterns)
```

- [ ] **Step 4: Run the test — verify it PASSES**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/poet_adapter/adapter.py poet_torch_huawei/tests_poet/test_adapter_unit.py
git commit -m "fix(huawei): dot-bound POET _name_matches so fc1 gate half isn't substring-excluded"
```

---

## Task 7: Turn the divisibility skip into a hard error (TDD)

**Files:**
- Modify: `poet_torch_huawei/megatron/core/poet_adapter/adapter.py:649-659` (native) and `:746-756` (TE)
- Modify: `poet_torch_huawei/tests_poet/test_adapter_unit.py` (add a case)

`_try_attach` reaches the divisibility check *before* any Megatron import (the `from megatron.core.tensor_parallel.layers import ...` lives in `install_poet_in_model`, not in `_try_attach`), and before any POET-state construction. A fake module with a real `nn.Parameter` weight + the size attributes drives the raise on CPU.

- [ ] **Step 1: Add the failing test**

Append to `poet_torch_huawei/tests_poet/test_adapter_unit.py`:
```python
import torch
import torch.nn as nn
from megatron.core.poet_adapter.adapter import _try_attach


class _FakeColumnLinear(nn.Module):
    """Minimal stand-in exposing the attrs _try_attach reads for kind='column'."""

    def __init__(self, out_local, in_local):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_local, in_local))
        self.output_size_per_partition = out_local
        self.input_size = in_local


def test_divisible_dims_attach_succeeds():
    m = _FakeColumnLinear(out_local=256, in_local=128)  # both divisible by 128
    ok = _try_attach(
        m, "decoder.layers.0.mlp.linear_fc1_gate", kind="column",
        block_size=128, normalize_weights=False, exclude_patterns=(),
    )
    assert ok is True
    assert getattr(m, "_poet_state", None) is not None


def test_indivisible_dims_hard_error():
    m = _FakeColumnLinear(out_local=200, in_local=128)  # 200 % 128 != 0
    with pytest.raises(RuntimeError, match="not divisible by block_size"):
        _try_attach(
            m, "decoder.layers.0.mlp.linear_fc1_gate", kind="column",
            block_size=128, normalize_weights=False, exclude_patterns=(),
        )
```

- [ ] **Step 2: Run — verify the indivisible case FAILS (currently returns False, no raise)**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v -k "divis or indivis"
```
Expected: `test_indivisible_dims_hard_error` FAILS (`DID NOT RAISE RuntimeError`); `test_divisible_dims_attach_succeeds` PASSES.

- [ ] **Step 3: Implement the hard error — native path**

In [adapter.py:649-659](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L649-L659), the current code is:
```python
    if out_local % block_size != 0 or in_local % block_size != 0:
        logger.info(
            "POET: skipping %s (kind=%s, out_local=%d, in_local=%d) -- "
            "not divisible by block_size=%d",
            module_name,
            kind,
            out_local,
            in_local,
            block_size,
        )
        return False
```
Replace with:
```python
    if out_local % block_size != 0 or in_local % block_size != 0:
        msg = (
            f"POET: cannot wrap {module_name} (kind={kind}, out_local={out_local}, "
            f"in_local={in_local}) -- not divisible by block_size={block_size}. "
            f"This layer is type/name eligible but its local dims don't tile into "
            f"{block_size}-blocks. Fix the dims, lower --poet-block-size, or exclude "
            f"this layer via --poet-exclude-modules / --poet-exclude-ancestors."
        )
        logger.error(msg)
        raise RuntimeError(msg)
```

- [ ] **Step 4: Implement the hard error — TE path**

In [adapter.py:746-756](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L746-L756), the current code is:
```python
    if out_local % block_size != 0 or in_local % block_size != 0:
        logger.info(
            "POET[TE]: skipping %s (%s, out_local=%d, in_local=%d) -- "
            "not divisible by block_size=%d",
            module_name,
            type(module).__name__,
            out_local,
            in_local,
            block_size,
        )
        return False
```
Replace with:
```python
    if out_local % block_size != 0 or in_local % block_size != 0:
        msg = (
            f"POET[TE]: cannot wrap {module_name} ({type(module).__name__}, "
            f"out_local={out_local}, in_local={in_local}) -- not divisible by "
            f"block_size={block_size}. This layer is type/name eligible but its "
            f"local dims don't tile into {block_size}-blocks. Fix the dims, lower "
            f"--poet-block-size, or exclude it via --poet-exclude-modules / "
            f"--poet-exclude-ancestors."
        )
        logger.error(msg)
        raise RuntimeError(msg)
```
(The `weight is None` and `out_local is None` guards above each stay soft `return False` — only the divisibility branch becomes a hard error, per the spec.)

- [ ] **Step 5: Run — verify all tests PASS**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v
```
Expected: all tests PASS (including both divisibility cases).

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/poet_adapter/adapter.py poet_torch_huawei/tests_poet/test_adapter_unit.py
git commit -m "feat(huawei): hard-error instead of silent skip on POET block-size indivisibility"
```

---

## Task 8: Single-GPU smoke — acceptance gate (on the `poet` node)

**Files:** none (validation only).

- [ ] **Step 1: Run the dev smoke (checkpoint saving off)**

Run (user, on the `poet` node):
```bash
SAVE_CKPT=0 codexlog poet_huawei_split_fc1 bash /lustre/fast/fast/zqiu/slm-research/scripts/train_poet_huawei.sh dev
```

- [ ] **Step 2: Verify acceptance criteria in `/lustre/home/zqiu/log/poet_huawei_split_fc1.log`**

1. **Router stays out** — no wrapped line whose module name ends in `.router` or `.gate`:
   ```bash
   grep -E "POET.*wrapped" /lustre/home/zqiu/log/poet_huawei_split_fc1.log | grep -iE "\.router|\.gate\b" || echo "ROUTER/GATE NOT WRAPPED ✓"
   ```
   Expected: `ROUTER/GATE NOT WRAPPED ✓`.
2. **Split took effect** — `linear_fc1_gate` and `linear_fc1_up` appear, and `linear_fc1` (fused) does not:
   ```bash
   grep -oE "wrapped [^ ]+" /lustre/home/zqiu/log/poet_huawei_split_fc1.log | grep -oE "linear_fc1(_gate|_up)?\b" | sort | uniq -c
   ```
   Expected: counts for `linear_fc1_gate` and `linear_fc1_up`, **zero** bare `linear_fc1`.
3. **Wrapped count = 100** — `[POET] ... wrapped 100 parallel-linear layers`. (Was 72: 16 attn + 28 fc1 + 28 fc2; now 16 + 56 gate/up + 28 fc2 = 100.)
4. **Step-0 loss matches the fused baseline** — first logged `lm loss` ≈ 11.97 (the prior fused-POET smoke's step-0). Cayley(0)=I ⇒ each half = exact `W₀`, so step 0 is numerically identical.
5. `[POET] merge-then-reinitialize at step 20`; 30/30 iterations; 0 NaN / 0 skipped; process exits 0 (no traceback).

- [ ] **Step 3: Triage if it fails (record which applied)**

- `AttributeError: 'MLP' object has no attribute 'linear_fc1'` → a caller hit `backward_dw`/`sharded_state_dict` on the split path; re-check Task 5 guards.
- POET wrapped count is 72 (unchanged) → `poet_split_fc1` didn't reach `MLP`; confirm Task 1 field + Task 2 arg line, and that `--use-poet` is set (it is, via the poet YAML).
- `RuntimeError: ... not divisible by block_size` at startup → a real divisibility problem now surfaced (Task 7 working as intended); check which layer/dim it names against `--poet-block-size 128`.
- Step-0 loss differs from 11.97 → the split forward isn't identity-equivalent; re-check Task 4 (`activation_func(gate) * up` ordering; gate is the first branch).

- [ ] **Step 4: Commit any dev-config/triage fix**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/
git commit -m "fix(huawei): split-fc1 smoke triage — <one line: what changed and why>"
```

---

## Task 9: Simplify the qkv split to the same config-only pattern (final, isolated)

Only start this **after** Task 8 passes. The vendored attention already reads `config.poet_split_qkv` in `__init__` and falls back to `submodules.linear_qkv` when `submodules.linear_q` is None ([attention.py:892-895](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/attention.py#L892-L895)). So the spec-level scaffolding in `gpt_layer_specs.py` is redundant: removing it leaves qkv purely config-driven (matching fc1) with identical behavior.

**Files:**
- Modify: `poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py:398-401`

- [ ] **Step 1: Revert the qkv spec branch to the plain fused emission**

In [gpt_layer_specs.py:398-401](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py#L398-L401), the current code is:
```python
                        linear_qkv=None if poet_split_qkv else backend.column_parallel_linear(),
                        linear_q=backend.column_parallel_linear() if poet_split_qkv else None,
                        linear_k=backend.column_parallel_linear() if poet_split_qkv else None,
                        linear_v=backend.column_parallel_linear() if poet_split_qkv else None,
```
Replace with the unconditional fused spec (attention's `__init__` does the split from config + the `linear_qkv` fallback):
```python
                        linear_qkv=backend.column_parallel_linear(),
```
Leave the `poet_split_qkv` parameter on the spec functions in place for now (harmless; removing the param threading across ~6 sites is out of scope and risks unrelated churn). The behavioral change is only this emission.

- [ ] **Step 2: Syntax check**

```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
python -m py_compile megatron/core/models/gpt/gpt_layer_specs.py && echo "gpt_layer_specs.py compiles"
```
Expected: `gpt_layer_specs.py compiles`.

- [ ] **Step 3: Re-run the smoke to confirm qkv still splits (user, `poet` node)**

```bash
SAVE_CKPT=0 codexlog poet_huawei_split_fc1_qkv bash /lustre/fast/fast/zqiu/slm-research/scripts/train_poet_huawei.sh dev
```
Verify in `/lustre/home/zqiu/log/poet_huawei_split_fc1_qkv.log`:
- `linear_q`, `linear_k`, `linear_v` still appear in the wrapped inventory (4 each), **zero** `linear_qkv`.
- Wrapped count still **100**; step-0 loss unchanged; merge at 20; clean exit.

- [ ] **Step 4: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py
git commit -m "refactor(huawei): drop redundant poet_split_qkv spec scaffolding (config-driven in attention)"
```

---

## Self-Review

**Spec coverage:**
- §5.1 config field → Task 1.
- §5.2 arg auto-on → Task 2.
- §5.3 MLPSubmodules fields → **intentionally dropped** (the "inside MLP" locus, decided with the user, builds both halves from the existing `submodules.linear_fc1`; no new submodule fields/spec threading needed). Documented in File Structure rationale.
- §5.4 spec emission → **dropped** for the same reason; replaced by Task 3 (`MLP.__init__` config-read).
- §5.5 MLP init/forward → Tasks 3 + 4.
- §5.6 `_name_matches` hardening → Task 6 (TDD).
- §5.7 divisibility hard error (both paths) → Task 7 (TDD).
- §6 checkpoint swiglu-factory guard → Task 5 (+ `backward_dw`, which the spec didn't enumerate but is required since split mode has no `self.linear_fc1`).
- §7 acceptance gates (router-out, split-effect, count 72→100, step-0 parity, merge/clean-exit, divisibility crash) → Task 8 (+ the CPU divisibility-raise test in Task 7).
- User decision: qkv cleanup as final isolated task → Task 9.

**Deviation from spec, with cause:** the spec's §5.3/§5.4 assumed spec-builder threading (mirroring qkv). During planning we confirmed the MoE specs live in a separate file (`moe_module_specs.py`) and routed-expert submodules come from a config-less `grouped_mlp_modules` — so the user chose the "inside MLP" locus, which produces the identical model graph from one file. The spec's intent (separate gate/up `ColumnParallelLinear`s, separate orbits, all three MLP kinds) is fully met; only the implementation locus changed. `experts.py` needs no edit because SequentialMLP routed experts serialize through `MLP.sharded_state_dict` (Task 5), and GroupedMLP (the only `experts.py` `sharded_state_dict`) is never the `MLP` class.

**Placeholder scan:** the only fill-in is Task 8 Step 4's `<one line>` commit reason — a human-authored description of an action taken, acceptable per the writing-plans convention (same as the prior huawei plan).

**Type/name consistency:** `split_fc1` attribute name is consistent across Tasks 3/4/5. `linear_fc1_gate` / `linear_fc1_up` used identically in Tasks 3 (build), 4 (forward), 5 (backward_dw), 6 (test names), 8 (acceptance grep). `_name_matches` and `_try_attach` signatures in the Task 6/7 tests match the real adapter signatures (`_try_attach(module, module_name, *, kind, block_size, normalize_weights, exclude_patterns, variant="poet", mem_efficient=False)`).

**Pre-commit note:** `poet_torch_huawei/` is excluded from slm-research's pre-commit hooks (recorded in the port's `.pre-commit-config.yaml`), so the vendored edits + new `tests_poet/` won't be reformatted/blocked. The `tests_poet/` dir is new source (not under the ignored `runs_dev/`/`__pycache__` globs), so it commits normally.
