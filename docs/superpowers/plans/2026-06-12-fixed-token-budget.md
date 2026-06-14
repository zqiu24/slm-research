# Fixed Token Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `training.total_tokens` to be specified explicitly (e.g. `500M`, `1B`, `10B`, `50B`, `100B`) so that near-scale architecture ablations share one identical, pre-built dataset, while keeping the existing `tokens_per_param × non_embedding_params` path as the default.

**Architecture:** Today `launchers/submit.py:resolve_config` (step 3) unconditionally computes `cfg.training.total_tokens = tokens_per_param * non_embedding_params`, and that flows into `--train-samples = total_tokens // seq_length` (`src/utils/megatron_args.py:_training_args`), which is part of Megatron's GPTDataset cache key. So two models that differ by a few params at the same nominal scale get different `--train-samples` → a slow dataset re-index AND slightly different data amounts. The fix is a precedence rule in token resolution — (1) decay-only resume `decay_tokens` (unchanged, highest), (2) explicit `training.total_tokens` (new), (3) `tokens_per_param` multiplier (default, unchanged) — plus a `parse_token_count()` helper that accepts `"500M"`/`"1B"`-style strings, and a `fixed_*` family of `training_regime` configs. No change is needed for the "switch dataset/tokenizer → rebuild but keep token count" requirement: `--data-cache-path` is already `runs/_data_cache/{data.name}` and the GPTDataset cache key includes the data path, so a different data axis rebuilds in its own cache dir while `--train-samples` (tokenizer-independent: `total_tokens // seq_length`) stays fixed. We pin that behavior with a test.

**Tech Stack:** Python 3.12, OmegaConf, pytest. Test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python` (base `python` lacks omegaconf/torch).

**Invariant to preserve:** the dataset identity is `(data.path, split, seed, seq_length, num_samples)`. Fixing `total_tokens` pins `num_samples` only when `seq_length` is also equal — true for same-rung comparisons (e.g. all ~60m scales use `seq_length: 256`). Document this; don't try to "fix" it.

**Run all test commands from the repo root:** `/lustre/fast/fast/zqiu/slm-research`.

---

### Task 1: `parse_token_count` helper in ladder_math

**Files:**
- Modify: `src/utils/ladder_math.py`
- Test: `tests/unit/test_ladder_math.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_ladder_math.py` (and add `parse_token_count` to the existing `from src.utils.ladder_math import (...)` block at the top):

```python
def test_parse_token_count_int_passthrough():
    assert parse_token_count(500_000_000) == 500_000_000


def test_parse_token_count_suffixes():
    assert parse_token_count("500M") == 500_000_000
    assert parse_token_count("1B") == 1_000_000_000
    assert parse_token_count("10B") == 10_000_000_000
    assert parse_token_count("50b") == 50_000_000_000
    assert parse_token_count("100B") == 100_000_000_000
    assert parse_token_count("2.5B") == 2_500_000_000
    assert parse_token_count("10K") == 10_000
    assert parse_token_count("1T") == 1_000_000_000_000


def test_parse_token_count_plain_strings():
    assert parse_token_count("1_000_000") == 1_000_000
    assert parse_token_count("5e9") == 5_000_000_000


@pytest.mark.parametrize("bad", ["", "abc", "10Q", "-1B", "B", 0, -5, 0.0])
def test_parse_token_count_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_token_count(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ladder_math.py -v`
Expected: ImportError — `cannot import name 'parse_token_count'`

- [ ] **Step 3: Implement `parse_token_count`**

Append to `src/utils/ladder_math.py`:

```python
_TOKEN_SUFFIXES = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def parse_token_count(value: int | float | str) -> int:
    """Parse an explicit token budget into an int token count.

    Accepts plain numbers (``500_000_000``, ``5e9``) and human-friendly
    suffixed strings (``"500M"``, ``"1B"``, ``"2.5b"``; K/M/B/T,
    case-insensitive). Underscores and commas in strings are ignored.
    Rejects non-positive results.
    """
    if isinstance(value, bool):
        raise ValueError(f"token count must be a number or suffixed string, got {value!r}")
    if isinstance(value, int | float):
        tokens = int(round(value))
    else:
        text = str(value).strip().replace("_", "").replace(",", "")
        if not text:
            raise ValueError(f"token count is empty: {value!r}")
        multiplier = _TOKEN_SUFFIXES.get(text[-1].upper())
        if multiplier is not None:
            text = text[:-1]
        else:
            multiplier = 1
        try:
            tokens = int(round(float(text) * multiplier))
        except ValueError as exc:
            raise ValueError(f"cannot parse token count {value!r}") from exc
    if tokens <= 0:
        raise ValueError(f"token count must be positive, got {value!r}")
    return tokens
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ladder_math.py -v`
Expected: all PASS (including the pre-existing ladder-math tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ladder_math.py tests/unit/test_ladder_math.py
git commit -m "$(cat <<'EOF'
feat(ladder): parse_token_count helper for 500M/1B-style token budgets
EOF
)"
```

---

### Task 2: Explicit `training.total_tokens` precedence in `resolve_config`

**Files:**
- Modify: `launchers/submit.py` (import block ~line 24; step 3 of `resolve_config`, currently lines 143–155)
- Test: `tests/unit/test_train_megatron_command.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_train_megatron_command.py` (it already imports `_parse_overrides` and `resolve_config`):

```python
def test_explicit_total_tokens_overrides_tokens_per_param():
    # ablation_20x sets tokens_per_param: 20, but an explicit budget must win,
    # including human-friendly "10B"-style strings from the CLI dotlist.
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
            "training.total_tokens=10B",
        ]
    )
    resolve_config(cfg)
    assert int(cfg.training.total_tokens) == 10_000_000_000


def test_default_path_still_uses_tokens_per_param():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)
    # 20 tokens/param * 60M non-embedding params
    assert int(cfg.training.total_tokens) == 1_200_000_000


def test_decay_tokens_still_beats_explicit_total_tokens():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd_decay_only",
            "experiment=champion",
            "training_regime=final_wsd_decay_only",
            "cluster=h800_cn",
            "training.decay_tokens=1200000000",
            "training.stable_checkpoint_dir=/tmp/stable_ckpt",
            "training.total_tokens=10B",
        ]
    )
    resolve_config(cfg)
    assert int(cfg.training.total_tokens) == 1_200_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_train_megatron_command.py -v`
Expected: `test_explicit_total_tokens_overrides_tokens_per_param` FAILS with an assertion error (`1200000000 != 10000000000`) — today `resolve_config` overwrites the explicit value with `20 * 60_000_000`. The other two new tests pass already (behavior unchanged); they are regression pins.

- [ ] **Step 3: Implement the precedence rule**

In `launchers/submit.py`, extend the ladder_math import (currently line 24):

```python
from src.utils.ladder_math import parse_token_count, total_tokens
```

Replace step 3 of `resolve_config` (the `# 3. Token count` block, currently lines 143–155):

```python
    # 3. Token count — precedence: decay-only resume (decay_tokens) > explicit
    # training.total_tokens (fixed-budget regimes / CLI override; accepts
    # "500M"/"1B"-style strings) > tokens_per_param * non_embedding_params.
    # An explicit budget pins --train-samples (= total_tokens // seq_length)
    # and therefore the GPTDataset cache key, so near-scale architecture
    # ablations share one pre-built dataset and see the same amount of data.
    if bool(cfg.training.get("resume_from_stable_stage", False)):
        decay_tokens = cfg.training.get("decay_tokens", None)
        if decay_tokens is None:
            raise ValueError(
                "resume_from_stable_stage requires training.decay_tokens "
                "(e.g. training.decay_tokens=1_200_000_000)"
            )
        cfg.training.total_tokens = int(decay_tokens)
    elif cfg.training.get("total_tokens", None) is not None:
        cfg.training.total_tokens = parse_token_count(cfg.training.total_tokens)
    else:
        cfg.training.total_tokens = total_tokens(
            cfg.training.tokens_per_param, int(cfg.base.non_embedding_params)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_train_megatron_command.py -v`
Expected: all PASS (including the pre-existing `test_decay_only_resolves_total_tokens_from_decay_tokens`)

- [ ] **Step 5: Commit**

```bash
git add launchers/submit.py tests/unit/test_train_megatron_command.py
git commit -m "$(cat <<'EOF'
feat(launcher): explicit training.total_tokens overrides tokens_per_param
EOF
)"
```

---

### Task 3: `megatron_args` fallback parity + dataset-pinning tests

`_training_args` has its own fallback for configs that never went through `resolve_config` (e.g. direct `build_megatron_args(cfg)` in tests). It already prefers `training.total_tokens` via `int(...) or ...`, but `int("10B")` would crash and `None` would crash. Make it use the same parser, and pin the two behaviors the user cares about: (a) same budget across near-scales → identical `--train-samples`; (b) different data axis → same `--train-samples`, different data path and cache dir (rebuild happens, token count fixed).

**Files:**
- Modify: `src/utils/megatron_args.py` (import block ~line 10; `_training_args`, currently lines 165–176)
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_total_tokens_suffix_string_accepted_without_resolve():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    cfg.training.total_tokens = "1B"  # as a CLI dotlist would leave it
    args = _args_to_map(build_megatron_args(cfg))
    # 60m uses seq_length 256: 1B tokens -> 3_906_250 samples
    assert args["--train-samples"] == str(1_000_000_000 // 256)
    assert args["--lr-decay-samples"] == str(1_000_000_000 // 256)


def test_fixed_total_tokens_pins_train_samples_across_scales_and_data():
    def args_for(scale: str, data: str) -> dict:
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                f"base/scale={scale}",
                "experiment=champion",
                "training_regime=ablation_20x",
                "cluster=h800_cn",
                f"data={data}",
            ]
        )
        cfg.training.total_tokens = 1_000_000_000
        return _args_to_map(build_megatron_args(cfg))

    a = args_for("60m", "nemotron_cc_v2_llama31_8b")
    b = args_for("300m", "nemotron_cc_v2_llama31_8b")
    c = args_for("60m", "nemotron_cc_v2_scratch_qwen3")

    # (a) Same budget, near scales (both seq 256) -> identical sample count,
    # i.e. identical GPTDataset cache key -> no rebuild between the two.
    assert a["--train-samples"] == b["--train-samples"] == str(1_000_000_000 // 256)
    # (b) Different dataset/tokenizer -> rebuild (different path + cache dir),
    # but the token budget stays exactly as specified.
    assert c["--train-samples"] == a["--train-samples"]
    assert c["--data-path"] != a["--data-path"]
    assert c["--data-cache-path"] != a["--data-cache-path"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -v`
Expected: `test_total_tokens_suffix_string_accepted_without_resolve` FAILS with `ValueError: invalid literal for int() with base 10: '1B'`. The second test should PASS already (int path works) — it is a regression pin.

- [ ] **Step 3: Implement parser parity**

In `src/utils/megatron_args.py`, add to the import block (after `from omegaconf import ...`, next to the other `src.utils` imports):

```python
from src.utils.ladder_math import parse_token_count
```

Replace the `else:` branch of the token computation in `_training_args` (currently lines 173–176):

```python
    else:
        explicit_tokens = training.get("total_tokens", None)
        if explicit_tokens is not None:
            total_tokens = parse_token_count(explicit_tokens)
        else:
            total_tokens = int(training.get("tokens_per_param", 20)) * int(
                cfg.base.non_embedding_params
            )
```

(Behavior note: this also fixes the old edge where `total_tokens: 0`-falsy handling silently fell back; explicit-but-invalid budgets now raise instead.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py tests/unit/test_ngpt_megatron_args.py -v`
Expected: all PASS (`test_ngpt_megatron_args.py` exercises the `tokens_per_param`-only dict path — must stay green)

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "$(cat <<'EOF'
feat(args): honor explicit total_tokens (incl. 1B-style strings) in megatron args
EOF
)"
```

---

### Task 4: `fixed_*` training-regime configs

Reproducible fixed-budget regimes for the budgets the ladder needs: 500M, 1B, 10B, 50B, 100B. Mirrors `ablation_20x.yaml`'s batch/checkpoint settings; only the token rule differs.

**Files:**
- Create: `configs/training_regime/fixed_500m.yaml`
- Create: `configs/training_regime/fixed_1b.yaml`
- Create: `configs/training_regime/fixed_10b.yaml`
- Create: `configs/training_regime/fixed_50b.yaml`
- Create: `configs/training_regime/fixed_100b.yaml`
- Test: `tests/unit/test_launcher_config_composition.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_launcher_config_composition.py` (add `from launchers.submit import resolve_config` to the existing import from `launchers.submit`):

```python
def test_fixed_regime_total_tokens_is_scale_independent():
    def resolved(scale: str):
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                f"base/scale={scale}",
                "experiment=champion",
                "training_regime=fixed_1b",
                "cluster=h800_cn",
            ]
        )
        resolve_config(cfg)
        return cfg

    small, large = resolved("60m"), resolved("300m")
    assert int(small.training.total_tokens) == 1_000_000_000
    assert int(large.training.total_tokens) == 1_000_000_000
    assert small.training.tokens_per_param is None


def test_all_fixed_regimes_parse_to_expected_budgets():
    expected = {
        "fixed_500m": 500_000_000,
        "fixed_1b": 1_000_000_000,
        "fixed_10b": 10_000_000_000,
        "fixed_50b": 50_000_000_000,
        "fixed_100b": 100_000_000_000,
    }
    for regime, budget in expected.items():
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                "base/scale=60m",
                "experiment=champion",
                f"training_regime={regime}",
                "cluster=h800_cn",
            ]
        )
        resolve_config(cfg)
        assert int(cfg.training.total_tokens) == budget, regime
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_launcher_config_composition.py -v`
Expected: the two new tests FAIL with `FileNotFoundError: No such config: .../configs/training_regime/fixed_1b.yaml`.

KNOWN PRE-EXISTING FAILURE in this file (baselined 2026-06-12, before this plan):
`test_parse_overrides_loads_defaults_and_data_axis` asserts the default cluster
is `h800_cn`, but `configs/launch/config.yaml` now defaults to `b200_de`. It is
unrelated to this plan — do NOT fix it here and do NOT count it against your
changes. It stays failing in Step 4 too.

- [ ] **Step 3: Create the five regime configs**

`configs/training_regime/fixed_1b.yaml`:

```yaml
# @package _global_
# Fixed token budget: 1B tokens regardless of model size (architecture
# comparisons at a shared dataset). Pins --train-samples
# (= total_tokens // seq_length) and hence the GPTDataset cache key, so
# near-scale ablations at the same seq_length reuse one pre-built dataset and
# see exactly the same data. Switching the data axis (other tokenizer) still
# rebuilds into its own runs/_data_cache/<data.name> at this same budget.
training:
  tokens_per_param: null                 # disabled; the budget is fixed below
  total_tokens: 1_000_000_000
  global_batch_size: 1024                # seqs/step, FROZEN across ladder (was 4M tokens / 4096)
  micro_batch_size: 32                   # matches ablation_20x (4×B200 activation budget)

checkpointing:
  save_every_tokens: 2_000_000_000
  keep_last: 3
  save_stable_stage_final: true          # required for WSD decay-only reuse
```

The other four files are identical except for the comment's budget and `total_tokens`:
- `fixed_500m.yaml`: `total_tokens: 500_000_000`
- `fixed_10b.yaml`: `total_tokens: 10_000_000_000`
- `fixed_50b.yaml`: `total_tokens: 50_000_000_000`
- `fixed_100b.yaml`: `total_tokens: 100_000_000_000`

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_launcher_config_composition.py -v`
Expected: both new tests PASS; only the pre-existing `test_parse_overrides_loads_defaults_and_data_axis` failure remains (see Step 2)

- [ ] **Step 5: Commit**

```bash
git add configs/training_regime/fixed_*.yaml tests/unit/test_launcher_config_composition.py
git commit -m "$(cat <<'EOF'
feat(config): fixed-token-budget training regimes (500M/1B/10B/50B/100B)
EOF
)"
```

---

### Task 5: Documentation updates

**Files:**
- Modify: `SPEC.md` (token-derivation lines: ~406, ~448–455, ~517, ~1000)
- Modify: `configs/base/scale/60m.yaml` (comment, lines 13–15)
- Modify: `configs/base/scale/300m.yaml` (the matching "not just a label" comment, ~line 25)
- Modify: `configs/training_regime/ablation_20x.yaml` (comment, line 5)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: SPEC.md — document the precedence**

At line ~406, after the sentence ending "...determines the token count.", append:

```markdown
Exception: a regime (or CLI override) may set `training.total_tokens`
explicitly (`fixed_500m` … `fixed_100b`, or `training.total_tokens=10B`);
it then takes precedence over `tokens_per_param` so that near-scale
architecture ablations share one pre-built dataset. `decay_tokens`
(decay-only resume) still outranks both.
```

At line ~517, replace the single derivation sentence:

```markdown
4. **Token count derivation**: `training.total_tokens = training.tokens_per_param * base.non_embedding_params`.
```

with:

```markdown
4. **Token count derivation**: decay-only resume uses `training.decay_tokens`;
   otherwise an explicit `training.total_tokens` (fixed-budget regimes /
   overrides, `"1B"`-style strings accepted) is used verbatim; otherwise
   `training.total_tokens = training.tokens_per_param * base.non_embedding_params`.
```

At the regime example (~line 448–455) add one line after `tokens_per_param: 20`:

```yaml
  # OR pin the budget exactly: total_tokens: 1_000_000_000 (and tokens_per_param: null)
```

At the pseudocode (~line 998–1001), replace:

```python
    cfg.training.total_tokens = (
        cfg.training.tokens_per_param * cfg.base.non_embedding_params
    )
```

with:

```python
    if cfg.training.get("total_tokens") is not None:  # explicit fixed budget
        cfg.training.total_tokens = parse_token_count(cfg.training.total_tokens)
    else:
        cfg.training.total_tokens = (
            cfg.training.tokens_per_param * cfg.base.non_embedding_params
        )
```

- [ ] **Step 2: Scale-config comments**

In `configs/base/scale/60m.yaml` lines 13–15, extend the NOTE sentence so it ends:

```yaml
# ... -> --train-samples -> GPTDataset cache key, so the token budget scales
# with it. To pin the dataset across near-scale variants instead, use a
# fixed-budget regime (training_regime=fixed_1b etc.) or set
# training.total_tokens explicitly — it overrides tokens_per_param.
```

Make the equivalent edit to the matching comment in `configs/base/scale/300m.yaml` (~line 25). In `configs/training_regime/ablation_20x.yaml` line 5, extend the comment:

```yaml
  # Total tokens computed at launch: tokens_per_param * base.non_embedding_params
  # (unless training.total_tokens is set explicitly — that pins the budget exactly)
```

- [ ] **Step 3: CHANGELOG.md**

Add under `## Unreleased` (top of section), matching the existing entry style:

```markdown
### Added — fixed token budgets (dataset pinning across architectures)

- `training.total_tokens` can now be set explicitly (config or CLI override;
  `"500M"`/`"1B"`-style strings accepted via `parse_token_count`) and takes
  precedence over `tokens_per_param * non_embedding_params` (decay-only
  resume's `decay_tokens` still ranks highest). This pins `--train-samples`
  and hence the GPTDataset cache key, so near-scale architecture ablations
  share one pre-built dataset and identical data amounts. New regimes:
  `training_regime/fixed_{500m,1b,10b,50b,100b}`. Switching the data axis
  (different tokenizer) still rebuilds into its own `runs/_data_cache/<name>`
  at the same fixed budget.
```

- [ ] **Step 4: Sanity-check docs didn't break anything**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_launcher_config_composition.py tests/unit/test_megatron_args.py -q`
Expected: PASS apart from the pre-existing `test_parse_overrides_loads_defaults_and_data_axis` failure (config comments are load-bearing only as YAML comments)

- [ ] **Step 5: Commit**

```bash
git add SPEC.md CHANGELOG.md configs/base/scale/60m.yaml configs/base/scale/300m.yaml configs/training_regime/ablation_20x.yaml
git commit -m "$(cat <<'EOF'
docs: document explicit total_tokens precedence and fixed-budget regimes
EOF
)"
```

---

### Task 6: Full CPU verification

**Files:** none (verification only)

- [ ] **Step 1: Compile check**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/utils/ladder_math.py src/utils/megatron_args.py launchers/submit.py`
Expected: exit 0, no output

- [ ] **Step 2: Full unit suite**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit -q`
Expected: green except KNOWN PRE-EXISTING failures baselined on 2026-06-12 before this plan (see "Pre-existing test failures" at the end of this document). Any failure NOT on that list must be treated as caused by this work — debug it before claiming success.

- [ ] **Step 3: End-to-end dry-run smoke (CPU-only, no GPU)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.submit base/family=llama3 base/scale=60m experiment=champion training_regime=fixed_1b cluster=h800_cn --dry-run
```
Expected: JSON output with `"total_tokens": 1000000000`. Then the same command with `training_regime=ablation_20x training.total_tokens=10B` → `"total_tokens": 10000000000`. (This writes a `runs/<run_name>/` archive dir — that is normal launcher behavior for dry runs.)

- [ ] **Step 4: Final commit (only if the dry-run archive or stragglers need adding — otherwise nothing to commit)**

```bash
git status --short   # confirm clean (runs/ archives are untracked scratch; leave them)
```

---

## Pre-existing test failures (baselined 2026-06-12, before any plan work)

- `tests/unit/test_launcher_config_composition.py::test_parse_overrides_loads_defaults_and_data_axis`
  — asserts the default cluster is `h800_cn`; `configs/launch/config.yaml` now
  defaults to `b200_de`. Stale test, unrelated to this plan. Leave failing;
  flag to the user at handoff.

(Targeted baseline: the six test files this plan touches or depends on —
`test_ladder_math`, `test_train_megatron_command`, `test_megatron_args`,
`test_launcher_config_composition`, `test_ngpt_megatron_args`,
`test_torchtitan_args` — ran 1 failed, 100 passed.)

---

## Out of scope (deliberately)

- **No change to GPTDataset/Megatron internals** — the cache-key behavior we rely on already exists.
- **No change to `src/utils/torchtitan_args.py`** — it reads `cfg.training.total_tokens` only after `resolve_config` has run (verified: all its callers and tests resolve first), so explicit budgets flow through it as plain ints automatically.
- **No fix for the pre-existing `test_parse_overrides_loads_defaults_and_data_axis` failure** (stale `h800_cn` default-cluster assertion vs. the current `b200_de` default) — unrelated to this feature; flag it to the user instead.
- **No seq_length normalization** — a fixed token budget pins the dataset only at equal `seq_length`; cross-seq-length comparisons are genuinely different datasets.
- **No warning when an explicit budget overrides `tokens_per_param`** — precedence is documented; the resolved config archive and the launcher's JSON output both show the final `total_tokens`.
- **No GPU/training runs** — this is config/launcher plumbing; CPU tests + dry-run fully cover it.
