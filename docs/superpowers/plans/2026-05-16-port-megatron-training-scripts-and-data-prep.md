# Amendment to the Megatron Runner and Data Port Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** This file used to contain a competing plan covering the same territory as [2026-05-16-megatron-runner-data-port.md](2026-05-16-megatron-runner-data-port.md). After reviewing both, the runner-data-port plan is canonical and supersedes this one for everything it covers. This file is now a small amendment carrying the corrections and optional follow-ups that did not fit into the canonical plan.

**Goal:** Patch a layer-freq / `num_layers` inconsistency in the canonical plan's DeepSeek path, fix the per-config data-cache directory, and add an optional end-to-end preprocessing driver. Three tightly scoped tasks; nothing here invents new architecture.

**Architecture:** Additive only. Adds one scale YAML, edits one line in the canonical plan's CLI builder, and adds one shell driver. Zero edits inside `third_party/Megatron-LM/`. Zero overlap with the canonical plan's translator, entrypoint, runner, optimizer routing, or train wrapper scripts — all of those are owned by [2026-05-16-megatron-runner-data-port.md](2026-05-16-megatron-runner-data-port.md).

**Tech Stack:** Python 3.12, OmegaConf, pytest, bash.

---

## Why this amendment exists

The canonical plan registers DeepSeek with `configs/base/family/deepseek_v3.yaml`, which carries `moe.layer_freq: "([0]*3+[1]*11)"`. That expression evaluates to a 14-element pattern (3 dense + 11 MoE). Megatron requires `len(layer_freq_eval) == num_layers`. The canonical plan's tests then exercise DeepSeek with `base/scale=600m`, which sets `num_layers=40`. At Megatron startup, that combination raises `AssertionError: moe-layer-freq must have len == num_layers (14 != 40)`. The `--dry-run` won't catch it — the bug lives in the resolved-but-not-yet-run command. We add a DeepSeek-shaped scale whose `num_layers` matches the family's `layer_freq` length, and we steer the wrapper script defaults to use it.

Second issue: the canonical plan's `_data_args()` sets `--data-cache-path runs/<config_hash>/data_cache`. Megatron's data cache caches the document-index for an indexed dataset, which is expensive to rebuild. Keying it on `config_hash` means every ablation rebuilds the index from scratch even when the dataset is identical. The cache should be keyed on dataset identity, not run identity.

Third (optional): the canonical plan ports three separate shell wrappers for preprocessing stages. A single driver that chains parquet → jsonl → tokenize is a small ergonomic add; not required.

## Prerequisites

- All eight tasks of [2026-05-16-megatron-runner-data-port.md](2026-05-16-megatron-runner-data-port.md) have landed. This plan amends paths that file creates; running it before that plan finishes is wrong order.

## File map

| Path | Action | Responsibility |
|---|---|---|
| [configs/base/scale/deepseek_v3_proxy_small.yaml](../../../configs/base/scale/deepseek_v3_proxy_small.yaml) | Create | DeepSeek-shaped scale whose `num_layers=14` matches the family's `layer_freq` length. |
| [src/utils/megatron_args.py](../../../src/utils/megatron_args.py) | Modify (one function) | Switch `--data-cache-path` from `runs/<config_hash>/data_cache` to `runs/_data_cache/<dataset_name>/`. |
| [scripts/train_adam.sh](../../../scripts/train_adam.sh) | Modify (one block) | When `--arch deepseek_v3`, default `base/scale` to `deepseek_v3_proxy_small`. Mirrored in `train_muon.sh` and `train_poet.sh`. |
| [scripts/train_muon.sh](../../../scripts/train_muon.sh) | Modify (one block) | Same default-scale handling. |
| [scripts/train_poet.sh](../../../scripts/train_poet.sh) | Modify (one block) | Same default-scale handling. |
| [tools/preprocess_nemotron_pipeline.sh](../../../tools/preprocess_nemotron_pipeline.sh) | Create | Optional convenience: chains the three single-stage wrappers added by the canonical plan. |
| [tests/unit/test_deepseek_proxy_small_scale.py](../../../tests/unit/test_deepseek_proxy_small_scale.py) | Create | Asserts the scale matches the family `layer_freq` length and the translator emits valid args. |
| [tests/unit/test_data_cache_path.py](../../../tests/unit/test_data_cache_path.py) | Create | Asserts `--data-cache-path` does NOT contain `config_hash` and DOES contain the dataset name. |

Three production files created, four modified. Two new tests. Nothing destructive.

---

### Task 1: Add the DeepSeek-shaped proxy-small scale

**Files:**
- Create: `configs/base/scale/deepseek_v3_proxy_small.yaml`
- Test: `tests/unit/test_deepseek_proxy_small_scale.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_deepseek_proxy_small_scale.py`:

```python
"""DeepSeek family + proxy-small scale must satisfy Megatron's layer_freq invariant.

The family YAML sets moe.layer_freq to a 14-element pattern. Megatron asserts
len(layer_freq_eval) == num_layers at startup. Pairing the family with the
right scale is what makes the dry-run-produced command actually launch.
"""
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.megatron_args import build_megatron_args


def _eval_layer_freq(expr: str) -> list[int]:
    """Evaluate Megatron's layer_freq mini-language. Restricted to lists / ints."""
    return list(eval(expr, {"__builtins__": {}}, {}))


def test_deepseek_proxy_small_num_layers_matches_family_layer_freq():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_proxy_small",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)

    pattern = _eval_layer_freq(str(cfg.base.model.moe.layer_freq))
    assert len(pattern) == int(cfg.base.model.num_layers), (
        f"layer_freq has {len(pattern)} elements but num_layers is "
        f"{cfg.base.model.num_layers}; Megatron will assert at startup."
    )


def test_deepseek_proxy_small_translator_emits_mla_and_moe_flags():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_proxy_small",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)

    args = build_megatron_args(cfg)
    assert "--multi-latent-attention" in args
    assert "--num-experts" in args
    i = args.index("--num-layers")
    assert args[i + 1] == "14"
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/unit/test_deepseek_proxy_small_scale.py -v`
Expected: FAIL with `FileNotFoundError: No such config: .../configs/base/scale/deepseek_v3_proxy_small.yaml`.

- [ ] **Step 3: Create the scale YAML**

Create `configs/base/scale/deepseek_v3_proxy_small.yaml`:

```yaml
# @package _global_
# DeepSeek-V3-shaped small proxy scale. num_layers=14 matches the family's
# moe.layer_freq pattern "([0]*3+[1]*11)" which evaluates to 14 elements.
# Megatron asserts len(layer_freq) == num_layers; pair this scale only
# with configs/base/family/deepseek_v3.yaml.
#
# Dimensions are scaled down from the upstream proxy test config
# (third_party/Megatron-LM/tests/functional_tests/test_cases/mixtral/
# deepseekv3_proxy_flex_tp1pp4emp16etp1cp1_release/model_config.yaml)
# by ~5x on hidden / FFN / heads to fit slm-research's ablation budget.
#
# non_embedding_params reports the ACTIVATED count (3 dense layers + 11
# MoE layers × topk-8 / num_experts-64), not the total. Parallelism rules
# in src/utils/parallelism.py are based on activation cost.
base:
  scale: "deepseek_v3_proxy_small"
  non_embedding_params: 1_500_000_000

  model:
    num_layers: 14
    hidden_size: 1536
    ffn_hidden_size: 3840
    num_attention_heads: 12
    num_query_groups: 12
    head_dim: 128
    kv_channels: 128
    seq_length: 4096
    max_position_embeddings: 4096
    tie_embeddings: false

    # Optional per-scale narrowing of family-level MoE so single-node EP
    # is feasible during ablations. Keep layer_freq's element count at 14;
    # shrinking the routed-expert count is the only safe override.
    moe:
      num_experts: 16
```

- [ ] **Step 4: Re-run the test**

Run: `pytest tests/unit/test_deepseek_proxy_small_scale.py -v`
Expected: PASS for both cases.

- [ ] **Step 5: Commit**

```bash
git add configs/base/scale/deepseek_v3_proxy_small.yaml tests/unit/test_deepseek_proxy_small_scale.py
git commit -m "scale: add deepseek_v3_proxy_small (14L, matches family layer_freq)"
```

---

### Task 2: Steer the optimizer wrapper scripts to the matching DeepSeek scale by default

The canonical plan's three wrapper scripts default to the global launch config's `base/scale` when the user doesn't override. That's `1_2b` for both Llama3 and DeepSeek-V3, which produces the layer_freq mismatch the moment the user passes `--arch deepseek_v3` without also passing `base/scale=...`. We add a small architecture-conditional default.

**Files:**
- Modify: `scripts/train_adam.sh`, `scripts/train_muon.sh`, `scripts/train_poet.sh`

- [ ] **Step 1: Update each wrapper to inject the architecture-default scale**

Each wrapper currently looks like (paraphrased from the canonical plan, Task 6 Step 3):

```bash
case "${ARCH}" in
  llama3) FAMILY="llama3" ;;
  deepseek_v3) FAMILY="deepseek_v3" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "experiment=optim/<name>" \
  "$@"
```

Replace the `case ... esac` and the trailing `python -m ...` block with:

```bash
case "${ARCH}" in
  llama3)
    FAMILY="llama3"
    DEFAULT_SCALE=""                # inherit launch config default (1_2b)
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

# Only inject the scale default if the user did not pass base/scale=...
USER_SET_SCALE="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "experiment=optim/<name>" \
  "$@"
```

In `train_adam.sh`, the experiment is `champion` (per the canonical plan). In `train_muon.sh`, `optim/muon_hybrid`. In `train_poet.sh`, `optim/poet`. Leave that line as-is for each script.

Note: under `set -u`, `"${SCALE_ARGS[@]}"` expands to zero arguments when the array is empty (this is the correct bash idiom — `"${arr[@]}"` is empty-safe; `"${arr[@]:-}"` is NOT).

- [ ] **Step 2: Re-run the canonical plan's script smoke tests against DeepSeek**

The canonical plan defines `tests/unit/test_train_scripts.py`. It tests DeepSeek under `train_muon.sh`. With the canonical scale `1_2b` that test would surface a layer_freq mismatch the moment the runner tries to compose the actual command. Verify the test still passes after our change:

```bash
pytest tests/unit/test_train_scripts.py -v
```

Expected: 3 passed (canonical tests). Add this DeepSeek-specific assertion to the existing `test_muon_script_supports_deepseek`:

```python
    # Amendment: confirm we steered to the matching scale.
    assert "base/scale=deepseek_v3_proxy_small" in proc.stdout or \
           '"--num-layers", "14"' in proc.stdout
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh tests/unit/test_train_scripts.py
git commit -m "scripts: default deepseek_v3 arch to proxy_small scale"
```

---

### Task 3: Fix the data-cache path so ablations share the indexed-dataset cache

Megatron's data-cache path is where it materialises the per-dataset document index (`doc-idx`, `sample-idx`, `shuffle-idx`). Building it is expensive (~minutes for a 2T-token corpus) and depends only on the dataset prefix + tokenizer + split + seq_length. Keying the cache on `config_hash` invalidates the cache for every architecture / optimizer / seed change, which is the wrong granularity.

**Files:**
- Modify: `src/utils/megatron_args.py` (one function, two lines)
- Test: `tests/unit/test_data_cache_path.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_data_cache_path.py`:

```python
"""Data cache must be keyed on dataset identity, not run identity."""
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.megatron_args import build_megatron_args


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_data_cache_path_is_dataset_keyed_not_config_keyed():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=1_2b",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
            "data=nemotron_cc_v2_llama31_8b",
        ]
    )
    resolve_config(cfg)

    args = build_megatron_args(cfg)
    cache = _arg_value(args, "--data-cache-path")

    assert str(cfg._derived.config_hash) not in cache, \
        "data cache must NOT be keyed on config_hash"
    assert cfg.data.name in cache, \
        f"data cache must reference dataset name; got {cache!r}"


def test_data_cache_is_stable_across_optim_changes():
    common = [
        "base/family=llama3",
        "base/scale=1_2b",
        "training_regime=ablation_20x",
        "cluster=h800_cn",
        "data=nemotron_cc_v2_llama31_8b",
    ]
    a = _parse_overrides([*common, "experiment=champion"])
    b = _parse_overrides([*common, "experiment=optim/muon_hybrid"])
    resolve_config(a)
    resolve_config(b)

    cache_a = _arg_value(build_megatron_args(a), "--data-cache-path")
    cache_b = _arg_value(build_megatron_args(b), "--data-cache-path")
    assert cache_a == cache_b, "switching optimiser must NOT invalidate data cache"
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/unit/test_data_cache_path.py -v`
Expected: FAIL — current code produces `runs/<config_hash>/data_cache`, so the path contains the hash and changes when the experiment changes.

- [ ] **Step 3: Edit `src/utils/megatron_args.py`**

In `_data_args()`, replace the line

```python
    _add(args, "--data-cache-path", f"runs/{cfg._derived.config_hash}/data_cache")
```

with

```python
    _add(args, "--data-cache-path", f"runs/_data_cache/{cfg.data.name}")
```

Rationale: cache identity must be a function of the data axis, not the run. `runs/_data_cache/<name>` keeps caches grouped under `runs/` (already in `.gitignore`) and re-uses one cache for every experiment / optimizer / seed against the same dataset.

- [ ] **Step 4: Re-run the test**

Run: `pytest tests/unit/test_data_cache_path.py tests/unit/test_megatron_args.py -v`
Expected: PASS for the new file; the canonical plan's `test_megatron_args.py` still passes (none of its existing assertions reference the cache path).

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_data_cache_path.py
git commit -m "args: key Megatron data cache on dataset name, not config_hash"
```

---

### Task 4 (optional): Add an end-to-end Nemotron preprocessing driver

Convenience only — the canonical plan already ports the three single-stage wrappers. This driver chains them so producing a new tokenized prefix is one command instead of three. Skip if not useful.

**Files:**
- Create: `tools/preprocess_nemotron_pipeline.sh`

- [ ] **Step 1: Create the driver**

Create `tools/preprocess_nemotron_pipeline.sh`:

```bash
#!/usr/bin/env bash
# End-to-end Nemotron preprocessing driver. Chains:
#   1. parquet -> jsonl (per-shard or whole-dir)
#   2. cat per-shard jsonl files into one
#   3. tokenize jsonl -> Megatron mmap (.bin/.idx)
#
# Stages may be skipped with --skip-stage {1|2|3} (repeatable).
#
# Each stage delegates to the wrappers added by the canonical plan
# (Task 7 of 2026-05-16-megatron-runner-data-port.md). This file does
# not duplicate their logic; it only orders them and gates them on flags.
set -euo pipefail

INPUT_DIR=""
JSONL_DIR=""
JSONL_MERGED=""
OUTPUT_PREFIX=""
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL=""
WORKERS="8"
IDX=""
SKIPS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --input-dir DIR              parquet shards (stage 1 input)
  --jsonl-dir DIR              per-shard jsonl out / cat input
  --jsonl-merged FILE          concatenated jsonl out / tokenize in
  --output-prefix PREFIX       .bin/.idx output prefix (no extension)
  --tokenizer-type NAME        (default: HuggingFaceTokenizer)
  --tokenizer-model PATH       HF model dir / SentencePiece .model
  --workers N                  (default: 8)
  --idx N                      stage-1 parquet shard index (optional)
  --skip-stage {1|2|3}         skip that stage (repeatable)
  -h | --help                  this text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)        INPUT_DIR="$2"; shift 2 ;;
    --jsonl-dir)        JSONL_DIR="$2"; shift 2 ;;
    --jsonl-merged)     JSONL_MERGED="$2"; shift 2 ;;
    --output-prefix)    OUTPUT_PREFIX="$2"; shift 2 ;;
    --tokenizer-type)   TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model)  TOKENIZER_MODEL="$2"; shift 2 ;;
    --workers)          WORKERS="$2"; shift 2 ;;
    --idx)              IDX="$2"; shift 2 ;;
    --skip-stage)       SKIPS+=("$2"); shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

want_stage() {
  local s="$1"
  for x in "${SKIPS[@]:-}"; do [[ "$x" == "$s" ]] && return 1; done
  return 0
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"

if want_stage 1; then
  [[ -n "$INPUT_DIR" && -n "$JSONL_DIR" ]] || \
    { echo "stage 1 needs --input-dir and --jsonl-dir" >&2; exit 2; }
  mkdir -p "$JSONL_DIR"
  echo "[stage 1] parquet -> jsonl"
  INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$JSONL_DIR" \
    bash tools/preprocess_nemotron_parquet_to_jsonl.sh ${IDX:-}
fi

if want_stage 2; then
  [[ -n "$JSONL_DIR" && -n "$JSONL_MERGED" ]] || \
    { echo "stage 2 needs --jsonl-dir and --jsonl-merged" >&2; exit 2; }
  echo "[stage 2] cat $JSONL_DIR/*.jsonl -> $JSONL_MERGED"
  : > "$JSONL_MERGED"
  for f in "$JSONL_DIR"/*.jsonl; do cat "$f" >> "$JSONL_MERGED"; done
fi

if want_stage 3; then
  [[ -n "$JSONL_MERGED" && -n "$OUTPUT_PREFIX" && -n "$TOKENIZER_MODEL" ]] || \
    { echo "stage 3 needs --jsonl-merged, --output-prefix, --tokenizer-model" >&2; exit 2; }
  mkdir -p "$(dirname "$OUTPUT_PREFIX")"
  echo "[stage 3] tokenize -> ${OUTPUT_PREFIX}.{bin,idx}"
  INPUT_FILE="$JSONL_MERGED" \
  OUTPUT_PREFIX="$OUTPUT_PREFIX" \
  TOKENIZER_TYPE="$TOKENIZER_TYPE" \
  TOKENIZER_MODEL="$TOKENIZER_MODEL" \
  WORKERS="$WORKERS" \
    bash tools/preprocess_nemotron_tokenize.sh
fi

echo "[done]"
```

- [ ] **Step 2: Make executable + syntax-check**

```bash
chmod +x tools/preprocess_nemotron_pipeline.sh
bash -n tools/preprocess_nemotron_pipeline.sh
./tools/preprocess_nemotron_pipeline.sh --help
```
Expected: exit 0; `--help` prints the usage block.

- [ ] **Step 3: Commit**

```bash
git add tools/preprocess_nemotron_pipeline.sh
git commit -m "tools: add end-to-end nemotron preprocessing driver"
```

---

## Self-review

- **Spec coverage:** Each task targets one issue called out in "Why this amendment exists". DeepSeek scale → Task 1. Wrapper default → Task 2. Data cache → Task 3. End-to-end driver → optional Task 4.
- **Placeholder scan:** No TBDs, no "fill in later". The wrapper-script patch in Task 2 references an existing test by name (`test_muon_script_supports_deepseek`) and the exact assertion to append.
- **Type consistency:** `cfg.data.name` is used in both Task 3's implementation and its test, matches the canonical plan's `configs/data/<name>.yaml` schema (`data.name: <stem>`). The scale field names (`num_layers`, `hidden_size`, `head_dim`, etc.) match the existing scale YAMLs (`configs/base/scale/{300m,600m,1_2b,2_4b,7b}.yaml`).
- **Order of operations:** Tasks 1-3 are independent and can be executed in any order. Task 4 has no dependencies. All four assume the canonical plan has fully landed.
