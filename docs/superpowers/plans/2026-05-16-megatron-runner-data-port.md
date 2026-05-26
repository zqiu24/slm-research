# Megatron Runner and Data Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add slm-research-native training wrappers for AdamW, Muon, and POET on Llama-3 and DeepSeek-style model families, plus slm-research data preprocessing wrappers and data configs, without editing `third_party/Megatron-LM/`.

**Architecture:** Megatron stays pinned and read-only under `third_party/Megatron-LM/`. slm-research owns the run composition, data catalog, command generation, optimizer adapters, and monkey patches under `configs/`, `launchers/`, `scripts/`, `src/`, and `tools/`. Runtime patches are applied inside every `torchrun` rank by a slm-research entrypoint before Megatron starts parsing args or building the model; the launcher records only the deterministic `patch_set_hash`.

**Tech Stack:** Python 3.12, PyTorch, Megatron-Core 0.17.0 at `9539a12e1b04a68423f57b3eb41d6125161dca24`, OmegaConf, pytest, shell wrappers, Megatron indexed dataset tools, `pyarrow` and `tqdm` for parquet-to-jsonl conversion.

---

## Context Map

- Source fork `/lustre/fast/fast/zqiu/Megatron-LM` is on `afe443bc4254762e4d91031bbfd074b6ba531d15`; its useful local additions are data-prep wrappers under `tools/` and Llama training scripts.
- Source fork `/lustre/scratch/zqiu/Megatron-LM` is on branch `poet_core_v0.16.1` at `bb43fa063a8fd77e40c53f963a66f3743fccac53`; it contains the POET optimizer, POET layer replacement helper, and Llama AdamW/Muon/POET scripts.
- Target `slm-research/third_party/Megatron-LM` already contains upstream Muon support in Megatron-Core 0.17.0, including `megatron/core/optimizer/muon.py`, `--optimizer muon`, `--optimizer dist_muon`, and Muon CLI args. Do not copy Muon into Megatron; add a thin `src.optim.muon` adapter only for slm-research tests and local API symmetry.
- Target `slm-research` already has a started POET port in `src/optim/poet.py`, `src/optim/poet_layers.py`, `src/patches/poet_*`, `third_party/poet_torch/`, and POET unit tests. The missing pieces are launcher/runtime wiring and the optimizer-setup patch that routes POET without adding `--optimizer poet` to Megatron.
- Preprocessed fast dataset prefix exists at `/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_text_document_llama31_8b` with a 2.3T `.bin` and 16G `.idx`.
- Scratch dataset prefixes exist at `/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_llama31_tokenizer_text_document`, `/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_qwen3_tokenizer_text_document`, and `/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_qwen35_tokenizer_text_document`.

## File Structure

- Create `configs/data/*.yaml` for concrete indexed dataset prefixes and tokenizer metadata.
- Create `configs/base/family/deepseek_v3.yaml`; reuse existing `configs/base/family/llama3.yaml`.
- Modify `launchers/submit.py` so `_parse_overrides()` composes default axes and axis override files correctly, including `experiment=optim/poet`, `cluster=h800_cn`, and `data=<name>`.
- Create `src/utils/megatron_args.py` as a pure, unit-tested config-to-Megatron-CLI translator.
- Create `launchers/pretrain_gpt_slm.py` as the per-rank entrypoint that applies slm-research patches and calls Megatron's GPT pretrain function.
- Create `launchers/train_megatron.py` as the parent launcher that resolves config, archives metadata, builds the `torchrun` command, and either prints it on `--dry-run` or runs it.
- Create `scripts/train_adam.sh`, `scripts/train_muon.sh`, and `scripts/train_poet.sh`; each accepts `llama3` or `deepseek_v3` as the first optional argument and forwards remaining overrides.
- Create `tools/preprocess_parquet_to_jsonl.py`, `tools/preprocess_nemotron_parquet_to_jsonl.sh`, `tools/preprocess_nemotron_tokenize.sh`, and `tools/preprocess_nemotron_merge.sh`.
- Modify `src/patches/_registry.py` to support hash-only patch metadata for launch archives, while keeping side-effect application for training ranks.
- Create `src/patches/poet_optimizer_setup.py` to route `--slm-optimizer poet` through `src.optim.poet.get_megatron_poet_optimizer`.
- Create `src/optim/muon.py` as a lazy adapter around Megatron's pinned Muon builder.

---

### Task 1: Config Composition and Data Catalog

**Files:**
- Modify: `launchers/submit.py`
- Modify: `configs/launch/config.yaml`
- Create: `configs/data/nemotron_cc_v2_llama31_8b.yaml`
- Create: `configs/data/nemotron_cc_v2_scratch_llama31.yaml`
- Create: `configs/data/nemotron_cc_v2_scratch_qwen3.yaml`
- Create: `configs/data/nemotron_cc_v2_scratch_qwen35.yaml`
- Test: `tests/unit/test_launcher_config_composition.py`

- [ ] **Step 1: Write the failing composition tests**

```python
# tests/unit/test_launcher_config_composition.py
"""Tests for slm-research config-axis composition."""

from __future__ import annotations

from pathlib import Path

import yaml

from launchers.submit import _parse_overrides


def test_parse_overrides_loads_defaults_and_data_axis():
    cfg = _parse_overrides([])

    assert cfg.base.family == "qwen3"
    assert cfg.base.scale == "1_2b"
    assert cfg.experiment.name == "champion"
    assert cfg.cluster.name == "h800_cn"
    assert cfg.data.name == "nemotron_cc_v2_llama31_8b"
    assert cfg.data.path == (
        "/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/"
        "nemotron_cc_v2_high_quality_text_document_llama31_8b"
    )


def test_parse_overrides_loads_nested_experiment_value():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet",
            "training_regime=ablation_40x",
            "cluster=h100_de",
            "data=nemotron_cc_v2_scratch_qwen3",
            "seed=7",
        ]
    )

    assert cfg.base.family == "llama3"
    assert cfg.base.scale == "600m"
    assert cfg.experiment.name == "poet"
    assert cfg.training.tokens_per_param == 40
    assert cfg.cluster.name == "h100_de"
    assert cfg.data.name == "nemotron_cc_v2_scratch_qwen3"
    assert cfg.seed == 7


def test_data_catalog_prefixes_point_at_bin_and_idx_files():
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "configs/data").glob("*.yaml")):
        data = yaml.safe_load(path.read_text())["data"]
        prefix = Path(data["path"])
        assert not str(prefix).endswith(".bin")
        assert not str(prefix).endswith(".idx")
        assert Path(str(prefix) + ".bin").exists(), f"{path} points at missing bin"
        assert Path(str(prefix) + ".idx").exists(), f"{path} points at missing idx"
```

- [ ] **Step 2: Run the tests and confirm the current failure**

Run:

```bash
pytest tests/unit/test_launcher_config_composition.py -v
```

Expected: failures showing default axes are not composed and `configs/data/` does not exist.

- [ ] **Step 3: Create the concrete data configs**

```yaml
# configs/data/nemotron_cc_v2_llama31_8b.yaml
# @package _global_
data:
  name: nemotron_cc_v2_llama31_8b
  path: "/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_text_document_llama31_8b"
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: "/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B"
  vocab_size: 128256
  split: "99,1,0"
  no_mmap_bin_files: true
  no_create_attention_mask_in_dataloader: true
  num_workers: 6
  expected_manifest_hash: null
```

```yaml
# configs/data/nemotron_cc_v2_scratch_llama31.yaml
# @package _global_
data:
  name: nemotron_cc_v2_scratch_llama31
  path: "/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_llama31_tokenizer_text_document"
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: "/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B"
  vocab_size: 128256
  split: "99,1,0"
  no_mmap_bin_files: true
  no_create_attention_mask_in_dataloader: true
  num_workers: 6
  expected_manifest_hash: null
```

```yaml
# configs/data/nemotron_cc_v2_scratch_qwen3.yaml
# @package _global_
data:
  name: nemotron_cc_v2_scratch_qwen3
  path: "/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_qwen3_tokenizer_text_document"
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: "/lustre/fast/fast/zqiu/hf_models/Qwen3-30B-A3B"
  vocab_size: 151936
  split: "99,1,0"
  no_mmap_bin_files: true
  no_create_attention_mask_in_dataloader: true
  num_workers: 6
  expected_manifest_hash: null
```

```yaml
# configs/data/nemotron_cc_v2_scratch_qwen35.yaml
# @package _global_
data:
  name: nemotron_cc_v2_scratch_qwen35
  path: "/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_qwen35_tokenizer_text_document"
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: "/lustre/fast/fast/zqiu/hf_models/Qwen3.5-35B-A3B-FP8"
  vocab_size: 151936
  split: "99,1,0"
  no_mmap_bin_files: true
  no_create_attention_mask_in_dataloader: true
  num_workers: 6
  expected_manifest_hash: null
```

- [ ] **Step 4: Add `data` to the default launch axes**

Replace the top of `configs/launch/config.yaml` with:

```yaml
# @package _global_
# Top-level Hydra-like entry config. `python -m launchers.submit` composes
# the axes below, then applies CLI overrides.
defaults:
  - base/family: qwen3
  - base/scale: 1_2b
  - experiment: champion
  - training_regime: ablation_20x
  - cluster: h800_cn
  - data: nemotron_cc_v2_llama31_8b
  - _self_

seed: 42

wandb:
  entity: "neckariumai-research"
  project: "sandbox-${oc.env:USER,unknown}"
  job_type: "sandbox"

allow_dirty: false

_derived: {}
```

- [ ] **Step 5: Replace `_parse_overrides()` with explicit axis composition**

In `launchers/submit.py`, add these helpers near `REPO_ROOT`:

```python
AXIS_TO_CONFIG_DIR = {
    "base/family": "configs/base/family",
    "base/scale": "configs/base/scale",
    "experiment": "configs/experiments",
    "training_regime": "configs/training_regime",
    "cluster": "configs/clusters",
    "data": "configs/data",
}


def _axis_config_path(axis: str, value: str) -> Path:
    if axis not in AXIS_TO_CONFIG_DIR:
        raise ValueError(f"Unknown config axis {axis!r}")
    path = REPO_ROOT / AXIS_TO_CONFIG_DIR[axis] / f"{value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No such config: {path}")
    return path


def _default_axis_entries(defaults: list) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for item in defaults:
        if item == "_self_":
            continue
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Unsupported defaults entry: {item!r}")
        axis, value = next(iter(item.items()))
        entries.append((str(axis), str(value)))
    return entries
```

Replace `_parse_overrides()` with:

```python
def _parse_overrides(pairs: list[str]) -> DictConfig:
    base = OmegaConf.load(REPO_ROOT / "configs/launch/config.yaml")
    defaults = list(base.pop("defaults", []) or [])

    axis_values = dict(_default_axis_entries(defaults))
    dotlist: list[str] = []

    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Bad override {raw!r}; expected KEY=VALUE")
        key, value = raw.split("=", 1)
        if key in AXIS_TO_CONFIG_DIR or key in {"base/family", "base/scale"}:
            axis_values[key] = value
        else:
            dotlist.append(raw)

    merges = [OmegaConf.load(_axis_config_path(axis, value)) for axis, value in axis_values.items()]
    resolved = OmegaConf.merge(*merges, base)
    if dotlist:
        resolved = OmegaConf.merge(resolved, OmegaConf.from_dotlist(dotlist))
    return resolved  # type: ignore[return-value]
```

- [ ] **Step 6: Run the composition tests**

Run:

```bash
pytest tests/unit/test_launcher_config_composition.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add launchers/submit.py configs/launch/config.yaml configs/data tests/unit/test_launcher_config_composition.py
git commit -m "feat: add data catalog and config-axis composition"
```

---

### Task 2: DeepSeek Family Config and Megatron CLI Builder

**Files:**
- Create: `configs/base/family/deepseek_v3.yaml`
- Create: `src/utils/megatron_args.py`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing CLI-builder tests**

```python
# tests/unit/test_megatron_args.py
"""Unit tests for translating slm configs into Megatron CLI args."""

from __future__ import annotations

from omegaconf import OmegaConf

from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args


def _args_to_map(args: list[str]) -> dict[str, str | bool]:
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


def test_llama3_adam_args_include_dense_gqa_rope_and_data_prefix():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--use-mcore-models"] is True
    assert args["--num-layers"] == "40"
    assert args["--hidden-size"] == "1280"
    assert args["--group-query-attention"] is True
    assert args["--num-query-groups"] == "4"
    assert args["--position-embedding-type"] == "rope"
    assert args["--rotary-base"] == "500000"
    assert args["--tokenizer-type"] == "HuggingFaceTokenizer"
    assert args["--data-path"].endswith("nemotron_cc_v2_high_quality_text_document_llama31_8b")
    assert args["--optimizer"] == "adam"


def test_deepseek_args_include_mla_moe_and_deepseek_router_knobs():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=600m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--multi-latent-attention"] is True
    assert args["--q-lora-rank"] == "1536"
    assert args["--kv-lora-rank"] == "512"
    assert args["--qk-head-dim"] == "128"
    assert args["--qk-pos-emb-head-dim"] == "64"
    assert args["--v-head-dim"] == "128"
    assert args["--num-experts"] == "64"
    assert args["--moe-router-topk"] == "8"
    assert args["--moe-router-score-function"] == "sigmoid"
    assert args["--moe-router-enable-expert-bias"] is True
    assert args["--enable-experimental"] is True


def test_muon_args_use_megatron_muon_and_disable_dist_optimizer_overlap():
    cfg = _parse_overrides(["experiment=optim/muon_hybrid"])
    args = build_megatron_args(cfg)
    amap = _args_to_map(args)

    assert amap["--optimizer"] == "muon"
    assert "--use-distributed-optimizer" not in args
    assert "--overlap-grad-reduce" not in args
    assert "--overlap-param-gather" not in args
    assert amap["--muon-num-ns-steps"] == "5"
    assert amap["--muon-momentum"] == "0.95"


def test_poet_args_use_slm_optimizer_and_keep_megatron_optimizer_adam():
    cfg = _parse_overrides(["experiment=optim/poet"])
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--optimizer"] == "adam"
    assert args["--slm-optimizer"] == "poet"
    assert args["--poet"] is True
    assert args["--poet-block-size"] == "256"
    assert args["--poet-merge-period"] == "200"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest tests/unit/test_megatron_args.py -v
```

Expected: import failure for `src.utils.megatron_args` and missing DeepSeek family config.

- [ ] **Step 3: Add the DeepSeek family config**

```yaml
# configs/base/family/deepseek_v3.yaml
# @package _global_
base:
  family: deepseek_v3
  family_version: "v3_proxy"
  reference: "DeepSeek-V3 style MLA + MoE config from Megatron functional proxy"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-6
    activation: "SwiGLU"
    positional_encoding: "rope"
    rotary_base: 10000
    rotary_scaling: null
    qk_norm: true
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
    attention_backend: "flash"
    multi_latent_attention: true
    q_lora_rank: 1536
    kv_lora_rank: 512
    qk_head_dim: 128
    qk_pos_emb_head_dim: 64
    v_head_dim: 128
    rotary_scaling_factor: 40
    mscale: 1.0
    mscale_all_dim: 1.0
    mtp_num_layers: 1
    mtp_loss_scaling_factor: 0.1
    moe:
      enabled: true
      num_experts: 64
      layer_freq: "([0]*3+[1]*11)"
      ffn_hidden_size: 2048
      shared_expert_intermediate_size: 2048
      router_load_balancing_type: "seq_aux_loss"
      router_topk: 8
      token_dispatcher_type: "flex"
      enable_deepep: true
      router_pre_softmax: true
      grouped_gemm: true
      aux_loss_coeff: 1.0e-4
      router_group_topk: 4
      router_num_groups: 8
      router_topk_scaling_factor: 2.5
      router_score_function: "sigmoid"
      router_enable_expert_bias: true
      router_bias_update_rate: 1.0e-3
      router_dtype: "fp32"
      permute_fusion: true
  tokenizer:
    nominal_name: "deepseek-v3"
    nominal_vocab_size: 129280
```

- [ ] **Step 4: Implement the pure CLI builder**

```python
# src/utils/megatron_args.py
"""Translate resolved slm-research configs into Megatron GPT CLI args."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _truthy(value: Any) -> bool:
    return bool(value)


def _add(args: list[str], flag: str, value: Any | None = None) -> None:
    args.append(flag)
    if value is not None:
        args.append(str(value))


def _maybe_bool(args: list[str], flag: str, value: Any) -> None:
    if _truthy(value):
        _add(args, flag)


def _sequence(values: Iterable[str]) -> list[str]:
    return [str(v) for v in values]


def _model_args(cfg: DictConfig) -> list[str]:
    base = cfg.base
    model = base.model
    args: list[str] = []

    _add(args, "--use-mcore-models")
    _add(args, "--num-layers", model.num_layers)
    _add(args, "--hidden-size", model.hidden_size)
    _add(args, "--ffn-hidden-size", model.ffn_hidden_size)
    _add(args, "--num-attention-heads", model.num_attention_heads)
    if int(model.num_query_groups) != int(model.num_attention_heads):
        _add(args, "--group-query-attention")
        _add(args, "--num-query-groups", model.num_query_groups)
    _add(args, "--kv-channels", model.head_dim)
    _add(args, "--seq-length", model.seq_length)
    _add(args, "--max-position-embeddings", model.get("max_position_embeddings", model.seq_length))
    _add(args, "--position-embedding-type", model.positional_encoding)
    _add(args, "--rotary-base", model.rotary_base)
    _add(args, "--rotary-percent", 1.0)
    _add(args, "--attention-dropout", model.attention_dropout)
    _add(args, "--hidden-dropout", model.hidden_dropout)
    _add(args, "--normalization", model.normalization)
    _add(args, "--norm-epsilon", model.norm_epsilon)
    _add(args, "--init-method-std", model.init_method_std)
    _add(args, "--attention-backend", model.get("attention_backend", "fused"))
    _add(args, "--swiglu")
    _add(args, "--disable-bias-linear")
    if not bool(model.tie_embeddings):
        _add(args, "--untie-embeddings-and-output-weights")

    if bool(model.get("qk_norm", False)):
        _add(args, "--qk-layernorm")

    if bool(model.get("multi_latent_attention", False)):
        _add(args, "--multi-latent-attention")
        for key, flag in (
            ("q_lora_rank", "--q-lora-rank"),
            ("kv_lora_rank", "--kv-lora-rank"),
            ("qk_head_dim", "--qk-head-dim"),
            ("qk_pos_emb_head_dim", "--qk-pos-emb-head-dim"),
            ("v_head_dim", "--v-head-dim"),
            ("rotary_scaling_factor", "--rotary-scaling-factor"),
            ("mscale", "--mscale"),
            ("mscale_all_dim", "--mscale-all-dim"),
            ("mtp_num_layers", "--mtp-num-layers"),
            ("mtp_loss_scaling_factor", "--mtp-loss-scaling-factor"),
        ):
            _add(args, flag, model[key])
        _add(args, "--enable-experimental")

    moe = model.get("moe", {})
    if bool(moe.get("enabled", False)):
        _add(args, "--num-experts", moe.num_experts)
        _add(args, "--moe-layer-freq", moe.layer_freq)
        _add(args, "--moe-ffn-hidden-size", moe.ffn_hidden_size)
        _add(args, "--moe-shared-expert-intermediate-size", moe.shared_expert_intermediate_size)
        _add(args, "--moe-router-load-balancing-type", moe.router_load_balancing_type)
        _add(args, "--moe-router-topk", moe.router_topk)
        _add(args, "--moe-token-dispatcher-type", moe.token_dispatcher_type)
        _maybe_bool(args, "--moe-enable-deepep", moe.enable_deepep)
        _maybe_bool(args, "--moe-router-pre-softmax", moe.router_pre_softmax)
        _maybe_bool(args, "--moe-grouped-gemm", moe.grouped_gemm)
        _add(args, "--moe-aux-loss-coeff", moe.aux_loss_coeff)
        _add(args, "--moe-router-group-topk", moe.router_group_topk)
        _add(args, "--moe-router-num-groups", moe.router_num_groups)
        _add(args, "--moe-router-topk-scaling-factor", moe.router_topk_scaling_factor)
        _add(args, "--moe-router-score-function", moe.router_score_function)
        _maybe_bool(args, "--moe-router-enable-expert-bias", moe.router_enable_expert_bias)
        _add(args, "--moe-router-bias-update-rate", moe.router_bias_update_rate)
        _add(args, "--moe-router-dtype", moe.router_dtype)
        _maybe_bool(args, "--moe-permute-fusion", moe.permute_fusion)

    return args


def _training_args(cfg: DictConfig) -> list[str]:
    training = cfg.training
    optim = cfg.optim
    model = cfg.base.model
    global_batch_tokens = int(training.global_batch_size_tokens)
    seq_length = int(model.seq_length)
    global_batch_size = global_batch_tokens // seq_length
    micro_batch_size = int(training.get("micro_batch_size", min(64, global_batch_size)))

    args: list[str] = []
    _add(args, "--micro-batch-size", micro_batch_size)
    _add(args, "--global-batch-size", global_batch_size)
    _add(args, "--train-samples", int(training.total_tokens) // seq_length)
    _add(args, "--lr-decay-samples", int(training.total_tokens) // seq_length)
    _add(args, "--lr-warmup-samples", max(1, (int(training.total_tokens) // seq_length) // 500))
    _add(args, "--lr", optim.get("lr", optim.get("adam", {}).get("lr", 1.0e-3)))
    _add(args, "--min-lr", training.get("min_lr", 1.0e-5))
    _add(args, "--lr-decay-style", training.get("lr_decay_style", "cosine"))
    _add(args, "--clip-grad", training.get("clip_grad", 1.0))
    _add(args, "--weight-decay", optim.get("weight_decay", 0.1))
    _add(args, "--bf16")
    _add(args, "--cross-entropy-loss-fusion")
    _add(args, "--calculate-per-token-loss")
    return args


def _optimizer_args(cfg: DictConfig) -> list[str]:
    optim = cfg.optim
    kind = str(optim.type)

    if kind == "adamw":
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
                "--slm-optimizer",
                "adamw",
            ]
        )

    if kind == "muon_hybrid":
        muon = optim.muon
        return _sequence(
            [
                "--optimizer",
                "muon",
                "--slm-optimizer",
                "muon",
                "--muon-momentum",
                optim.get("muon_momentum", 0.95),
                "--muon-num-ns-steps",
                muon.ns_steps,
                "--muon-scale-mode",
                optim.get("muon_scale_mode", "spectral"),
                "--muon-tp-mode",
                optim.get("muon_tp_mode", "blockwise"),
            ]
        )

    if kind == "poet":
        poet = optim.poet
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--slm-optimizer",
                "poet",
                "--poet",
                "--poet-block-size",
                poet.block_size,
                "--poet-init-type",
                poet.init_type,
                "--poet-mup-alpha",
                poet.mup_alpha,
                "--poet-merge-period",
                poet.merge_period,
                "--poet-scale",
                poet.scale,
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
            ]
        )

    raise ValueError(f"Unsupported optimizer type {kind!r}")


def _parallel_args(cfg: DictConfig) -> list[str]:
    args: list[str] = []
    _add(args, "--tensor-model-parallel-size", cfg.parallelism.tp)
    _add(args, "--pipeline-model-parallel-size", cfg.parallelism.pp)
    if bool(cfg.parallelism.get("sequence_parallel", True)):
        _add(args, "--sequence-parallel")
    if cfg.optim.type == "adamw" and bool(cfg.parallelism.get("distributed_optimizer", False)):
        _add(args, "--use-distributed-optimizer")
        _add(args, "--overlap-grad-reduce")
        _add(args, "--overlap-param-gather")
    return args


def _data_args(cfg: DictConfig) -> list[str]:
    data = cfg.data
    args: list[str] = []
    _add(args, "--data-path", data.path)
    _add(args, "--tokenizer-type", data.tokenizer_type)
    _add(args, "--tokenizer-model", data.tokenizer_model)
    _add(args, "--vocab-size", data.vocab_size)
    _add(args, "--data-cache-path", f"runs/{cfg._derived.config_hash}/data_cache")
    _add(args, "--split", data.split)
    if bool(data.no_mmap_bin_files):
        _add(args, "--no-mmap-bin-files")
    if bool(data.no_create_attention_mask_in_dataloader):
        _add(args, "--no-create-attention-mask-in-dataloader")
    _add(args, "--num-workers", data.num_workers)
    return args


def _logging_args(cfg: DictConfig) -> list[str]:
    archive = f"runs/{cfg._derived.config_hash}"
    return _sequence(
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
            f"{archive}/checkpoints",
            "--wandb-project",
            cfg.wandb.project,
            "--wandb-exp-name",
            f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s{cfg.seed}",
        ]
    )


def build_megatron_args(cfg: DictConfig) -> list[str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    args: list[str] = []
    args.extend(_model_args(cfg))
    args.extend(_training_args(cfg))
    args.extend(_optimizer_args(cfg))
    args.extend(_parallel_args(cfg))
    args.extend(_data_args(cfg))
    args.extend(_logging_args(cfg))
    return args
```

- [ ] **Step 5: Run the CLI-builder tests**

Run:

```bash
pytest tests/unit/test_megatron_args.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add configs/base/family/deepseek_v3.yaml src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat: translate slm configs to Megatron CLI args"
```

---

### Task 3: Patch Hashing and POET Optimizer Runtime Wiring

**Files:**
- Modify: `src/patches/_registry.py`
- Modify: `src/patches/__init__.py`
- Modify: `launchers/submit.py`
- Create: `src/patches/poet_optimizer_setup.py`
- Modify: `configs/experiments/optim/poet.yaml`
- Test: `tests/unit/test_patches_registry.py`
- Test: `tests/unit/test_patch_poet_optimizer_setup.py`

- [ ] **Step 1: Extend the registry tests for hash-only behavior**

Append to `tests/unit/test_patches_registry.py`:

```python
def test_patch_set_hash_does_not_apply_registered_patch():
    from src.patches._registry import _REGISTRY, patch_set_hash, register_patch

    calls = []

    @register_patch(name="hash_only_example", targets=("pkg.fn",))
    def apply():
        calls.append("applied")

    h = patch_set_hash(["hash_only_example"])

    assert len(h) == 16
    assert calls == []
    assert _REGISTRY["hash_only_example"].applied is False
```

- [ ] **Step 2: Run the registry tests and confirm failure**

Run:

```bash
pytest tests/unit/test_patches_registry.py -v
```

Expected: `ImportError` or `AttributeError` for missing `patch_set_hash`.

- [ ] **Step 3: Add hash-only helpers to the registry**

In `src/patches/_registry.py`, add:

```python
def patch_set_hash(names: list[str] | tuple[str, ...]) -> str:
    """Return the deterministic hash for registered patches without applying them."""
    names = sorted(set(names))
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise UnknownPatch(f"Unknown patches: {unknown}. Registered: {sorted(_REGISTRY)}")
    if not names:
        return "noop" + "0" * 12
    payload = "\n".join(f"{n}:{_REGISTRY[n].source_sha}" for n in names)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
```

Then replace the last payload block in `apply_patches()` with:

```python
    return patch_set_hash(names)
```

In `src/patches/__init__.py`, export `patch_set_hash` alongside `apply_patches`.

- [ ] **Step 4: Update launcher resolution to hash patches without relying on parent-process side effects**

In `launchers/submit.py`, import `patch_set_hash` from `src.patches` and replace `_apply_experiment_patches()` with:

```python
def _register_experiment_patches(cfg: DictConfig) -> str:
    """Import patch modules named by cfg and return their deterministic hash.

    Training ranks apply the patches inside launchers.pretrain_gpt_slm. The
    parent launcher only records the hash, so dry-runs do not mutate Megatron.
    """
    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    for name in patches:
        importlib.import_module(f"src.patches.{name}")
    return patch_set_hash(patches)
```

Then in `resolve_config()`, replace:

```python
cfg._derived.patch_set_hash = _apply_experiment_patches(cfg)
```

with:

```python
cfg._derived.patch_set_hash = _register_experiment_patches(cfg)
```

Update existing tests in `tests/unit/test_launcher_patch_wiring.py` to import `_register_experiment_patches` instead of `_apply_experiment_patches`; keep the existing assertions about registered patch names and hash length.

- [ ] **Step 5: Write the failing POET optimizer setup patch test**

```python
# tests/unit/test_patch_poet_optimizer_setup.py
"""Tests for the POET optimizer setup patch."""

from __future__ import annotations

import importlib
import sys
import types

from src.patches._registry import _reset_for_tests


def test_poet_optimizer_setup_registers_targets():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)

    importlib.import_module("src.patches.poet_optimizer_setup")

    from src.patches import registered_patches

    entry = registered_patches()["poet_optimizer_setup"]
    assert "megatron.training.training.get_megatron_optimizer_config" in entry.targets
    assert "megatron.training.training.get_megatron_optimizer" in entry.targets


def test_poet_optimizer_setup_routes_slm_optimizer_to_poet_builder(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    calls = []

    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1.0e-3)
        return cfg, {"from": "original"}

    def original_get_optimizer(config, model, **kwargs):
        calls.append(("original", config, model, kwargs))
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_builder = types.SimpleNamespace()

    def fake_poet_builder(config, model_chunks, **kwargs):
        calls.append(("poet", config, model_chunks, kwargs))
        return "poet-optimizer"

    fake_builder.get_megatron_poet_optimizer = fake_poet_builder

    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)
    monkeypatch.setitem(sys.modules, "src.optim.poet", fake_builder)

    patch_mod.apply()

    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=200,
        poet_scale=1.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
    )
    cfg, overrides = fake_training.get_megatron_optimizer_config(args)
    assert overrides == {"from": "original"}
    assert cfg.slm_optimizer == "poet"
    assert cfg.poet_merge_period == 200
    assert cfg.poet_scale == 1.5

    out = fake_training.get_megatron_optimizer(cfg, ["model"], use_gloo_process_groups=False)
    assert out == "poet-optimizer"
    assert calls[-1][0] == "poet"
```

- [ ] **Step 6: Run the POET patch test and confirm failure**

Run:

```bash
pytest tests/unit/test_patch_poet_optimizer_setup.py -v
```

Expected: missing module `src.patches.poet_optimizer_setup`.

- [ ] **Step 7: Implement `poet_optimizer_setup`**

```python
# src/patches/poet_optimizer_setup.py
"""Patch: route slm-research POET optimizer through Megatron's Adam branch.

Targets:
- megatron.training.training.get_megatron_optimizer_config
- megatron.training.training.get_megatron_optimizer

Megatron-Core 0.17.0 does not parse `--optimizer poet`. slm-research passes
`--optimizer adam --slm-optimizer poet` and this patch attaches the POET
settings to the OptimizerConfig, then routes the optimizer builder call to
`src.optim.poet.get_megatron_poet_optimizer`.
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)


@register_patch(name="poet_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "poet":
            return config, overrides
        config.slm_optimizer = "poet"
        config.poet_merge_period = getattr(args, "poet_merge_period", 0)
        config.poet_scale = getattr(args, "poet_scale", 1.0)
        config.poet_block_size = getattr(args, "poet_block_size", 256)
        config.poet_init_type = getattr(args, "poet_init_type", "normalized")
        config.poet_mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "poet":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.poet import get_megatron_poet_optimizer

        return get_megatron_poet_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
```

- [ ] **Step 8: Add the patch to the POET experiment**

In `configs/experiments/optim/poet.yaml`, insert `poet_optimizer_setup` before the model patches:

```yaml
  patches:
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
```

- [ ] **Step 9: Run the patch tests**

Run:

```bash
pytest tests/unit/test_patches_registry.py tests/unit/test_launcher_patch_wiring.py tests/unit/test_patch_poet_optimizer_setup.py -v
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/patches/_registry.py src/patches/__init__.py launchers/submit.py \
        src/patches/poet_optimizer_setup.py configs/experiments/optim/poet.yaml \
        tests/unit/test_patches_registry.py tests/unit/test_launcher_patch_wiring.py \
        tests/unit/test_patch_poet_optimizer_setup.py
git commit -m "feat: hash patches at launch and route poet optimizer at runtime"
```

---

### Task 4: Muon Adapter for slm-research API Symmetry

**Files:**
- Create: `src/optim/muon.py`
- Modify: `src/optim/__init__.py`
- Test: `tests/unit/test_muon_adapter.py`
- Modify: `configs/experiments/optim/muon_hybrid.yaml`

- [ ] **Step 1: Write the failing Muon adapter tests**

```python
# tests/unit/test_muon_adapter.py
"""Unit tests for the slm-research Muon adapter."""

from __future__ import annotations

import sys
import types


def test_muon_adapter_lazy_routes_to_megatron_builder(monkeypatch):
    from src.optim import muon as muon_mod

    calls = []

    fake_muon_module = types.SimpleNamespace()

    def fake_get_megatron_muon_optimizer(config, model_chunks, **kwargs):
        calls.append((config, model_chunks, kwargs))
        return "muon-optimizer"

    fake_muon_module.get_megatron_muon_optimizer = fake_get_megatron_muon_optimizer
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.muon", fake_muon_module)

    cfg = types.SimpleNamespace()
    out = muon_mod.get_megatron_muon_optimizer(
        cfg,
        ["model"],
        config_overrides={"x": 1},
        use_gloo_process_groups=False,
        layer_wise_distributed_optimizer=False,
    )

    assert out == "muon-optimizer"
    assert calls[0][0] is cfg
    assert calls[0][1] == ["model"]
    assert calls[0][2]["config_overrides"] == {"x": 1}
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
pytest tests/unit/test_muon_adapter.py -v
```

Expected: missing module `src.optim.muon`.

- [ ] **Step 3: Implement the adapter**

```python
# src/optim/muon.py
"""Thin adapter around Megatron-Core's pinned Muon optimizer.

Muon is already present in `third_party/Megatron-LM` at the slm-research pin.
This module gives slm-research a stable import surface without copying Muon
implementation code into `src/`.
"""

from __future__ import annotations

from typing import Any


def get_megatron_muon_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Any = None,
) -> Any:
    from megatron.core.optimizer.muon import (
        get_megatron_muon_optimizer as _get_megatron_muon_optimizer,
    )

    return _get_megatron_muon_optimizer(
        config=config,
        model_chunks=model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        layer_wise_distributed_optimizer=layer_wise_distributed_optimizer,
        pg_collection=pg_collection,
    )
```

- [ ] **Step 4: Add current Megatron Muon knobs to the config schema**

In `src/optim/__init__.py`, add fields to `OptimizerCfg`:

```python
    muon_momentum: float = 0.95
    muon_num_ns_steps: int = 5
    muon_scale_mode: str = "spectral"
    muon_tp_mode: str = "blockwise"
    muon_extra_scale_factor: float = 1.0
    muon_coefficient_type: str = "quintic"
    muon_scalar_optimizer: str = "adam"
```

In `configs/experiments/optim/muon_hybrid.yaml`, add runtime knobs under `optim`:

```yaml
  muon_momentum: 0.95
  muon_scale_mode: spectral
  muon_tp_mode: blockwise
  muon_extra_scale_factor: 1.0
  muon_coefficient_type: quintic
  muon_scalar_optimizer: adam
```

- [ ] **Step 5: Run optimizer tests**

Run:

```bash
pytest tests/unit/test_muon_adapter.py tests/unit/test_optim_dispatch.py tests/unit/test_megatron_args.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/optim/muon.py src/optim/__init__.py configs/experiments/optim/muon_hybrid.yaml \
        tests/unit/test_muon_adapter.py
git commit -m "feat: expose pinned Megatron Muon through slm optimizer adapter"
```

---

### Task 5: Per-rank Megatron Entrypoint and Parent Runner

**Files:**
- Create: `launchers/pretrain_gpt_slm.py`
- Create: `launchers/train_megatron.py`
- Test: `tests/unit/test_pretrain_gpt_slm.py`
- Test: `tests/unit/test_train_megatron_command.py`

- [ ] **Step 1: Write the failing entrypoint extra-args test**

```python
# tests/unit/test_pretrain_gpt_slm.py
"""Tests for slm-research's Megatron GPT entrypoint helpers."""

from __future__ import annotations

import argparse

from launchers.pretrain_gpt_slm import add_slm_args


def test_add_slm_args_accepts_poet_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)

    args = parser.parse_args(
        [
            "--slm-config-hash",
            "abc123",
            "--slm-optimizer",
            "poet",
            "--poet",
            "--poet-block-size",
            "256",
            "--poet-init-type",
            "normalized",
            "--poet-mup-alpha",
            "1.0",
            "--poet-merge-period",
            "200",
            "--poet-scale",
            "1.5",
        ]
    )

    assert args.slm_config_hash == "abc123"
    assert args.slm_optimizer == "poet"
    assert args.poet is True
    assert args.poet_block_size == 256
    assert args.poet_merge_period == 200
    assert args.poet_scale == 1.5
```

- [ ] **Step 2: Write the failing parent-runner dry-run test**

```python
# tests/unit/test_train_megatron_command.py
"""Tests for parent torchrun command generation."""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_megatron import build_torchrun_command


def test_build_torchrun_command_targets_slm_entrypoint():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/poet"])
    resolve_config(cfg)

    cmd = build_torchrun_command(cfg)

    assert cmd[:3] == ["torchrun", "--nproc_per_node", str(cfg.cluster.gpus_per_node)]
    assert "-m" in cmd
    assert "launchers.pretrain_gpt_slm" in cmd
    assert "--slm-config-hash" in cmd
    assert str(cfg._derived.config_hash) in cmd
    assert "--slm-optimizer" in cmd
    assert "poet" in cmd
```

- [ ] **Step 3: Run the tests and confirm failure**

Run:

```bash
pytest tests/unit/test_pretrain_gpt_slm.py tests/unit/test_train_megatron_command.py -v
```

Expected: missing modules `launchers.pretrain_gpt_slm` and `launchers.train_megatron`.

- [ ] **Step 4: Implement `launchers/pretrain_gpt_slm.py`**

```python
# launchers/pretrain_gpt_slm.py
"""Per-rank Megatron GPT entrypoint for slm-research.

This module is launched by torchrun. It applies slm-research patches inside
the rank process, then calls Megatron's GPT pretrain function from the pinned
third_party checkout.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGATRON_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"


def add_slm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("slm-research")
    group.add_argument("--slm-config-hash", type=str, required=True)
    group.add_argument("--slm-optimizer", choices=["adamw", "muon", "poet"], default="adamw")
    group.add_argument("--poet", action="store_true")
    group.add_argument("--poet-block-size", type=int, default=256)
    group.add_argument("--poet-init-type", choices=["none", "normalized", "mup_normalized"], default="normalized")
    group.add_argument("--poet-mup-alpha", type=float, default=1.0)
    group.add_argument("--poet-merge-period", type=int, default=0)
    group.add_argument("--poet-scale", type=float, default=1.0)
    return parser


def _prepend_paths() -> None:
    for path in (REPO_ROOT, MEGATRON_ROOT):
        text = os.fspath(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_resolved_config(config_hash: str):
    path = REPO_ROOT / "runs" / config_hash / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Resolved config not found: {path}")
    return OmegaConf.load(path)


def _apply_runtime_patches(cfg) -> None:
    from src.patches import apply_patches

    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    for name in patches:
        importlib.import_module(f"src.patches.{name}")
    apply_patches(patches)


def _combined_extra_args_provider(existing_provider):
    def provider(parser):
        if existing_provider is not None:
            parser = existing_provider(parser)
        return add_slm_args(parser)

    return provider


def main() -> None:
    _prepend_paths()

    config_hash = None
    for idx, item in enumerate(sys.argv):
        if item == "--slm-config-hash" and idx + 1 < len(sys.argv):
            config_hash = sys.argv[idx + 1]
            break
    if config_hash is None:
        raise RuntimeError("--slm-config-hash must be present in torchrun args")

    cfg = _load_resolved_config(config_hash)
    _apply_runtime_patches(cfg)

    import pretrain_gpt as mg
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps

    set_startup_timestamps(
        program_start=mg._PROGRAM_START_TIME,
        main_entry=mg.time.time(),
    )
    mg.train_valid_test_datasets_provider.is_distributed = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        mg.train_valid_test_datasets_provider,
        partial(mg.model_provider, mg.gpt_builder),
        ModelType.encoder_or_decoder,
        mg.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=_combined_extra_args_provider(
            mg.add_modelopt_args if mg.has_nvidia_modelopt else None
        ),
        store=store,
        get_embedding_ranks=mg.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `launchers/train_megatron.py`**

```python
# launchers/train_megatron.py
"""Resolve slm config and launch Megatron through the slm per-rank entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from launchers.submit import REPO_ROOT, _parse_overrides, archive_resolved_config, resolve_config
from src.utils.megatron_args import build_megatron_args


def build_torchrun_command(cfg) -> list[str]:
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(cfg.cluster.gpus_per_node),
        "--nnodes",
        str(cfg.cluster.nodes or 1),
        "--node_rank",
        str(os.environ.get("NODE_RANK", "0")),
        "--master_addr",
        str(os.environ.get("MASTER_ADDR", "localhost")),
        "--master_port",
        str(os.environ.get("MASTER_PORT", "6000")),
        "-m",
        "launchers.pretrain_gpt_slm",
        "--slm-config-hash",
        str(cfg._derived.config_hash),
    ]
    cmd.extend(build_megatron_args(cfg))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _parse_overrides(args.overrides)
    resolve_config(cfg)
    archive = archive_resolved_config(cfg)
    cmd = build_torchrun_command(cfg)

    payload = {
        "config_hash": str(cfg._derived.config_hash),
        "archive": os.fspath(archive),
        "command": cmd,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.fspath(REPO_ROOT),
            os.fspath(Path(REPO_ROOT) / "third_party" / "Megatron-LM"),
            env.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the new launcher tests**

Run:

```bash
pytest tests/unit/test_pretrain_gpt_slm.py tests/unit/test_train_megatron_command.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add launchers/pretrain_gpt_slm.py launchers/train_megatron.py \
        tests/unit/test_pretrain_gpt_slm.py tests/unit/test_train_megatron_command.py
git commit -m "feat: launch Megatron through slm per-rank entrypoint"
```

---

### Task 6: Three Optimizer Run Scripts with Llama3 and DeepSeek Selection

**Files:**
- Create: `scripts/train_adam.sh`
- Create: `scripts/train_muon.sh`
- Create: `scripts/train_poet.sh`
- Test: `tests/unit/test_train_scripts.py`

- [ ] **Step 1: Write the failing script smoke tests**

```python
# tests/unit/test_train_scripts.py
"""Smoke tests for optimizer wrapper scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, arch: str):
    return subprocess.run(
        ["bash", f"scripts/{script}", arch, "--dry-run", "cluster.nodes=1", "cluster.gpus_per_node=1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_adam_script_supports_llama3():
    proc = _run("train_adam.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "base/family=llama3" not in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "adamw" in proc.stdout


def test_muon_script_supports_deepseek():
    proc = _run("train_muon.sh", "deepseek_v3")
    assert "--optimizer" in proc.stdout
    assert "muon" in proc.stdout
    assert "--multi-latent-attention" in proc.stdout


def test_poet_script_supports_llama3():
    proc = _run("train_poet.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout
    assert "poet" in proc.stdout
    assert "--poet-merge-period" in proc.stdout
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest tests/unit/test_train_scripts.py -v
```

Expected: `scripts/train_adam.sh` not found.

- [ ] **Step 3: Create a shared script pattern**

Use this full content for `scripts/train_adam.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3" ;;
  deepseek_v3) FAMILY="deepseek_v3" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "experiment=champion" \
  "$@"
```

Use this full content for `scripts/train_muon.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3" ;;
  deepseek_v3) FAMILY="deepseek_v3" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "experiment=optim/muon_hybrid" \
  "$@"
```

Use this full content for `scripts/train_poet.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3" ;;
  deepseek_v3) FAMILY="deepseek_v3" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "experiment=optim/poet" \
  "$@"
```

- [ ] **Step 4: Make scripts executable**

Run:

```bash
chmod +x scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh
```

- [ ] **Step 5: Run the script smoke tests**

Run:

```bash
pytest tests/unit/test_train_scripts.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh tests/unit/test_train_scripts.py
git commit -m "feat: add adam muon poet training wrappers"
```

---

### Task 7: Data Preprocessing Wrappers

**Files:**
- Modify: `pyproject.toml`
- Create: `tools/preprocess_parquet_to_jsonl.py`
- Create: `tools/preprocess_nemotron_parquet_to_jsonl.sh`
- Create: `tools/preprocess_nemotron_tokenize.sh`
- Create: `tools/preprocess_nemotron_merge.sh`
- Test: `tests/unit/test_preprocess_parquet_to_jsonl.py`

- [ ] **Step 1: Add preprocessing dependencies**

In `pyproject.toml`, add:

```toml
[project.optional-dependencies]
data = [
    "pyarrow>=15.0",
    "tqdm>=4.66",
]
```

If `[project.optional-dependencies]` already exists, merge the `data` group beside `dev` and `gpu`.

- [ ] **Step 2: Write the failing converter test**

```python
# tests/unit/test_preprocess_parquet_to_jsonl.py
"""Tests for parquet to jsonl preprocessing."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from tools.preprocess_parquet_to_jsonl import parquet_to_jsonl


def test_parquet_to_jsonl_filters_empty_text_and_writes_json(tmp_path):
    parquet_path = tmp_path / "part_000.parquet"
    table = pa.table({"text": [" hello ", "", None, "world"], "other": [1, 2, 3, 4]})
    pq.write_table(table, parquet_path)

    output = tmp_path / "out"
    parquet_to_jsonl(
        parquet_files=[parquet_path],
        output_dir=output,
        file_prefix="nemotron",
        text_column="text",
        batch_size=2,
        max_rows_per_file=10,
        output_path=None,
    )

    files = sorted(output.glob("nemotron_part_*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert rows == [{"text": "hello"}, {"text": "world"}]
```

- [ ] **Step 3: Run the test and confirm failure**

Run:

```bash
pytest tests/unit/test_preprocess_parquet_to_jsonl.py -v
```

Expected: missing `tools.preprocess_parquet_to_jsonl`.

- [ ] **Step 4: Port the parquet converter with ASCII logs and deterministic file order**

```python
# tools/preprocess_parquet_to_jsonl.py
"""Convert parquet text shards to Megatron-compatible JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


class RotatingJsonlWriter:
    def __init__(self, output_dir: Path, file_prefix: str, max_rows_per_file: int):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.max_rows = max_rows_per_file
        self.current_file_idx = 0
        self.current_file_rows = 0
        self.file_handle = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_next_file()

    def _open_next_file(self) -> None:
        if self.file_handle:
            self.file_handle.close()
        path = self.output_dir / f"{self.file_prefix}_part_{self.current_file_idx:05d}.jsonl"
        print(f"creating {path}")
        self.file_handle = path.open("w", encoding="utf-8")
        self.current_file_idx += 1
        self.current_file_rows = 0

    def write(self, row: dict[str, str]) -> None:
        self.file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.current_file_rows += 1
        if self.current_file_rows >= self.max_rows:
            self._open_next_file()

    def close(self) -> None:
        if self.file_handle:
            self.file_handle.close()


def parquet_to_jsonl(
    *,
    parquet_files: list[Path],
    output_dir: Path,
    file_prefix: str = "nemotron",
    text_column: str = "text",
    batch_size: int = 4096,
    max_rows_per_file: int = 100_000,
    output_path: Path | None = None,
) -> int:
    total_docs = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    handle = None
    if output_path is None:
        writer = RotatingJsonlWriter(output_dir, file_prefix, max_rows_per_file)
    else:
        handle = output_path.open("w", encoding="utf-8")

    try:
        for parquet_file in sorted(Path(p) for p in parquet_files):
            parquet = pq.ParquetFile(parquet_file)
            for batch in tqdm(
                parquet.iter_batches(batch_size=batch_size, columns=[text_column]),
                desc=f"reading {parquet_file.name}",
                total=parquet.num_row_groups,
            ):
                table = batch.to_pydict()
                for raw_text in table.get(text_column, []):
                    if raw_text is None:
                        continue
                    text = str(raw_text).strip()
                    if not text:
                        continue
                    row = {"text": text}
                    if handle is not None:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        writer.write(row)
                    total_docs += 1
    finally:
        if handle is not None:
            handle.close()
        if writer is not None:
            writer.close()
    print(f"processed_documents={total_docs}")
    return total_docs


def _resolve_input(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise RuntimeError(f"No parquet files found under {path}")
        return files
    if path.is_file():
        return [path]
    raise RuntimeError(f"Input path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="nemotron")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    files = _resolve_input(Path(args.input))
    output_path = None
    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(files):
            raise RuntimeError(f"Index {args.idx} out of range [0, {len(files) - 1}]")
        selected = files[args.idx]
        files = [selected]
        output_path = Path(args.output_dir) / f"{args.prefix}_{selected.stem}.jsonl"

    parquet_to_jsonl(
        parquet_files=files,
        output_dir=Path(args.output_dir),
        file_prefix=args.prefix,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_rows_per_file=args.max_rows,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add the three shell wrappers**

```bash
# tools/preprocess_nemotron_parquet_to_jsonl.sh
#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality_jsonl}"
IDX="${1:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${IDX}" ]]; then
  python -m tools.preprocess_parquet_to_jsonl --input "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" --idx "${IDX}"
else
  python -m tools.preprocess_parquet_to_jsonl --input "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
fi
```

```bash
# tools/preprocess_nemotron_tokenize.sh
#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_full.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_llama31_tokenizer}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B}"
WORKERS="${WORKERS:-8}"

python third_party/Megatron-LM/tools/preprocess_data.py \
  --input "${INPUT_FILE}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --tokenizer-type "${TOKENIZER_TYPE}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --workers "${WORKERS}" \
  --append-eod
```

```bash
# tools/preprocess_nemotron_merge.sh
#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality_processed}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/Nemotron-CC-v2-High-Quality_merged}"

python third_party/Megatron-LM/tools/merge_datasets.py \
  --input "${INPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}"
```

- [ ] **Step 6: Make wrappers executable**

Run:

```bash
chmod +x tools/preprocess_nemotron_parquet_to_jsonl.sh \
         tools/preprocess_nemotron_tokenize.sh \
         tools/preprocess_nemotron_merge.sh
```

- [ ] **Step 7: Run preprocessing tests**

Run:

```bash
pytest tests/unit/test_preprocess_parquet_to_jsonl.py -v
```

Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tools/preprocess_parquet_to_jsonl.py \
        tools/preprocess_nemotron_parquet_to_jsonl.sh \
        tools/preprocess_nemotron_tokenize.sh tools/preprocess_nemotron_merge.sh \
        tests/unit/test_preprocess_parquet_to_jsonl.py
git commit -m "feat: add Nemotron preprocessing wrappers"
```

---

### Task 8: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/tokenizer_policy.md`
- Create: `docs/data_preprocessing.md`
- Test: full selected test suite

- [ ] **Step 1: Update README run examples**

Replace the current script block with:

```markdown
```bash
# Llama-3 family
scripts/train_adam.sh llama3 --dry-run
scripts/train_muon.sh llama3 --dry-run
scripts/train_poet.sh llama3 --dry-run

# DeepSeek-V3-style family
scripts/train_adam.sh deepseek_v3 --dry-run
scripts/train_muon.sh deepseek_v3 --dry-run
scripts/train_poet.sh deepseek_v3 --dry-run

# Override any axis or scalar config inline
scripts/train_adam.sh llama3 base/scale=600m data=nemotron_cc_v2_scratch_qwen3 seed=7
```
```

- [ ] **Step 2: Write data preprocessing docs**

```markdown
# Data Preprocessing

Nemotron-CC-v2 preprocessing has two stages:

1. Convert parquet shards to JSONL with `tools/preprocess_nemotron_parquet_to_jsonl.sh`.
2. Tokenize JSONL into Megatron indexed dataset files with `tools/preprocess_nemotron_tokenize.sh`.

The default data config points at:

`/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_text_document_llama31_8b`

Scratch variants are registered under `configs/data/`:

- `nemotron_cc_v2_scratch_llama31`
- `nemotron_cc_v2_scratch_qwen3`
- `nemotron_cc_v2_scratch_qwen35`

Use a registered dataset from training scripts with:

```bash
scripts/train_adam.sh llama3 data=nemotron_cc_v2_scratch_qwen3 --dry-run
```

The tokenizer policy remains: compare optimizer and architecture runs only
within one data config unless the experiment is explicitly about tokenizers.
```

- [ ] **Step 3: Run all touched unit tests**

Run:

```bash
pytest \
  tests/unit/test_launcher_config_composition.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patches_registry.py \
  tests/unit/test_launcher_patch_wiring.py \
  tests/unit/test_patch_poet_optimizer_setup.py \
  tests/unit/test_muon_adapter.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_train_megatron_command.py \
  tests/unit/test_train_scripts.py \
  tests/unit/test_preprocess_parquet_to_jsonl.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 4: Run dry-run commands for both architecture families**

Run:

```bash
scripts/train_adam.sh llama3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
scripts/train_muon.sh llama3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
scripts/train_poet.sh llama3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
scripts/train_adam.sh deepseek_v3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
scripts/train_muon.sh deepseek_v3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
scripts/train_poet.sh deepseek_v3 --dry-run cluster.nodes=1 cluster.gpus_per_node=1
```

Expected: each prints JSON with `config_hash`, `archive`, and `command`; DeepSeek commands include `--multi-latent-attention`; POET commands include `--slm-optimizer poet` and a non-noop `patch_set_hash`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/tokenizer_policy.md docs/data_preprocessing.md
git commit -m "docs: document Megatron training and data preprocessing flow"
```

---

## Self-Review

- Spec coverage:
  - Three run scripts: Task 6 creates exactly `scripts/train_adam.sh`, `scripts/train_muon.sh`, and `scripts/train_poet.sh`.
  - Llama3 and DeepSeek architecture selection: Task 2 adds `deepseek_v3` and Task 6 accepts `llama3` or `deepseek_v3`.
  - Megatron read-only: all new runtime behavior is in `launchers/`, `scripts/`, `src/`, `tools/`, and `configs/`; no task edits `third_party/Megatron-LM/`.
  - Data preprocessing scripts: Task 7 ports parquet-to-jsonl and tokenization wrappers into slm-research.
  - Existing preprocessed data paths: Task 1 registers fast and scratch dataset prefixes.
- Placeholder scan:
  - The plan contains concrete paths, config values, commands, and expected outcomes.
  - `expected_manifest_hash: null` is intentional because manifest hashing is not available in the current repo; the launcher already records `unverified` when this field is null.
- Type consistency:
  - `cfg.data.*`, `cfg.base.model.*`, `cfg.optim.*`, and `cfg._derived.config_hash` names are consistent across tests, launcher code, and CLI builder code.
  - POET uses `--slm-optimizer poet` while keeping Megatron's native `--optimizer adam`, so Megatron's argparse choices stay unchanged.
