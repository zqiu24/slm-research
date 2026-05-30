# torchtitan training backend — design

**Date:** 2026-05-30
**Status:** approved (design v2 — goal refined 2026-05-30), pending implementation plan
**Scope:** add a *second* training backend (`backend=torchtitan`) that is
**interface-identical from the slm-research side** — same six-axis yaml, same
`scripts/train_*.sh`, same config structure, same `experiment`/`patches`
declarations. You flip `--backend torchtitan` and *nothing else changes*. Under
the hood it leans on torchtitan's **native** model classes and interfaces
(`llama3`, `qwen3`, `deepseek_v3`) as hard as possible. **Exact model-structure
parity with Megatron is an explicit non-goal** — the torchtitan model need only
*qualify as the same family*. First cut: AdamW, bf16, FSDP2 (+ expert-parallel
for `deepseek_v3`), sized to slm's scale ladder.

---

## Problem

slm-research has exactly one training backend: **Megatron-Core**, vendored as a
git submodule at `third_party/Megatron-LM` (pinned SHA, never edited — SPEC.md
§4.1). Every run flows through it:

```
scripts/train_*.sh
  └─ launchers/train_megatron.py        # resolve 6-axis Hydra cfg, archive resolved_config.yaml,
     │                                  #   build torchrun command
     └─ torchrun -m launchers.pretrain_gpt_slm  --slm-config-path ...  <megatron CLI flags>
        └─ Megatron pretrain()
```

The config→backend translation lives in `src/utils/megatron_args.py` (≈440
lines of `cfg → Megatron CLI flags`). Architecture variants are reached through
`src/specs/gpt_layer_specs.py` (ModuleSpec tree), optimizers through
`src/optim/` (Megatron-optimizer interface), and anything ModuleSpec can't
reach through SHA-keyed monkey-patches in `src/patches/`.

We want a **PyTorch-native** alternative backend — [torchtitan](https://github.com/pytorch/torchtitan)
— to exploit FSDP2/DTensor per-parameter sharding, `torch.compile`, native
torchao Float8/MXFP8, and — crucially — torchtitan's **first-party model zoo**.
At the v0.2.2 pin `torchtitan/models/` already ships `llama3`, `qwen3`,
`deepseek_v3` (MoE + MLA), `llama4`, `gpt_oss` and a shared `moe/`, each with its
own model class, `infra/parallelize.py`, and a flavor (size) registry. So the
torchtitan backend **never ports a model** — the single hardest part of the
Megatron backend (hand-built `ModuleSpec` trees + monkey-patches for MLA/MoE)
simply does not exist here. We reuse torchtitan's models through their own
interfaces and only register slm's *sizes* as new flavors.

The core difficulty is that torchtitan is opinionated about two things
slm-research already fixes elsewhere:

1. **Data.** torchtitan's default dataloader streams **C4 from HuggingFace and
   tokenizes on the fly**. slm-research trains on a frozen, pre-tokenized
   Megatron `.bin/.idx` corpus (Nemotron-CC-v2 HQ, Llama-3.1-8B tokenizer,
   2.4 TB). To honor the "same `data=` config feeds either backend" principle,
   the torchtitan backend must read that **same corpus** — done by reusing
   Megatron's already-vendored `GPTDataset` as a torchtitan dataloader. (Because
   model structure no longer has to match Megatron, byte-identical *order* is a
   convenient correctness check, not a hard reproducibility requirement.) This is
   still the #1 integration item.
2. **Config surface.** torchtitan is driven by a TOML `JobConfig` plus dotted
   CLI overrides (`torchrun -m torchtitan.train --job.config_file <toml>
   --training.global_batch_size ...`), not by a flat flag list. The new
   `cfg → torchtitan` mapper must emit that TOML + override pair — and must
   **warn-and-skip** any Megatron-only knob the native torchtitan model can't
   express, so the *same* experiment yaml runs on either backend.

## Goals

1. **Identical slm-research-side experience.** Same six-axis yaml, same
   `scripts/train_*.sh`, same config structure, same `experiment`/`patches`
   declarations. The *only* user-visible change between backends is
   `--backend torchtitan`. The config/structure handling on the slm side is
   unchanged from how Megatron is driven.
2. **A real `backend` axis.** A first-class `backend ∈ {megatron, torchtitan}`
   field (default `megatron`) on the resolved config, giving a torchtitan run a
   **distinct, non-colliding** run identity (run-name + run-dir) from its
   Megatron twin, recorded with `torchtitan_sha`.
3. **Use torchtitan's native code through its own interfaces.** Drive
   `torchtitan.train` with a generated TOML; reuse torchtitan's model classes,
   `TrainSpec`, parallelize/dataloader/optimizer builders. The vendored submodule
   is **never edited**; all slm wiring rides torchtitan's `experimental.custom_import`
   extension hook. Keep torchtitan's code logic intact — adapt the config to it,
   not it to the config.
4. **Multi-family from the torchtitan zoo.** Wire `base/family ∈ {llama3, qwen3,
   deepseek_v3}` straight onto torchtitan's first-party model of the same name,
   registering slm's scale ladder (300m/600m/1.2b/…) as new flavors. No model is
   ported or reimplemented.
5. **Same corpus via the same `data=` config.** The torchtitan backend reads the
   same Megatron `.bin/.idx` corpus by reusing the vendored `GPTDataset`, so the
   identical `data=` axis works on either backend.
6. **Graceful config superset.** Megatron-only knobs and `src/patches/` entries
   that torchtitan's native model can't express are **warn-and-skipped**, not
   errors — so one experiment yaml runs on both backends.
7. **Reproducibility plumbing.** Record `torchtitan_sha`; archive the generated
   TOML into the run dir; keep `--dry-run` (no-GPU) working for CI.

## Non-goals (explicitly out, or deferred to later named phases)

- **Exact model-structure / numerics parity with Megatron.** Tied-vs-untied
  embeddings, exact init, exact FFN width, and a bit-for-bit (or curve-for-curve)
  match are **explicitly out**. The torchtitan model only has to *qualify as the
  same family*. The functional gate is "trains and converges healthily," not
  "reproduces the Megatron curve." (An informal llama3 cross-check vs Megatron is
  welcome as a sanity signal, but gates nothing.)
- **Custom training algorithms** — POET, Muon/muon_hybrid, nGPT, sandwich-norm.
  AdamW is the first cut; the non-AdamW wrappers reject `--backend torchtitan`
  with a clear message until each is ported as its own follow-on.
- **Float8/MXFP8** in the first cut (bf16 first; Float8 is a mapped follow-on
  toggle via torchao).
- **Cross-backend checkpoint reuse.** torchtitan uses DCP; Megatron uses its own
  distributed-checkpoint format. They are **not** interchangeable; the WSD
  checkpoint-reuse story (SPEC.md §7.1) stays Megatron-only for now.
- **Pipeline parallel tuning.** Small-scale dense runs are FSDP2-only; TP/PP/CP
  are mapped but exercised later. `deepseek_v3` additionally uses torchtitan's
  native **expert parallel** as needed.

## Third-party pin

- **torchtitan, git submodule at `third_party/torchtitan`.**
- **Tag / SHA:** `v0.2.2` → `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`.
- All torchtitan releases are GitHub *pre-releases*; `v0.2.2` is the newest
  tagged version and is treated as "last stable" per the pin-to-a-tag contract
  used for Megatron-LM (SPEC.md §4.1).
- A new `docs/torchtitan_pin.md` records the SHA + bump procedure, mirroring
  `docs/megatron_pin.md`. A second `[submodule]` stanza is added to
  `.gitmodules`.

torchtitan v0.2.2 facts the design relies on (verified against the tag):
- Entrypoint: `torchrun -m torchtitan.train --job.config_file <toml>` (+ dotted
  overrides); `run_train.sh` is a thin wrapper we do **not** use.
- Config: TOML `JobConfig` + CLI overrides.
- Models: llama3 (8B/70B/405B configs ship; the `llama3` model + parallelize
  helpers are first-party). Smaller scales are just `[model]` dimension changes.
- Parallelism: FSDP2 (per-param sharding), HSDP/DDP, TP (+async), PP, CP — all
  via DeviceMesh.
- Dataloader: a **component** (`torchtitan/components/dataloader.py`) — custom
  datasets are registerable via the train-spec/component system; default is C4.
- Float8/MXFP8 via torchao (`torchtitan/components/quantization/`).

## Architecture

### Drive path (mirror Megatron — do not reinvent)

```
scripts/train_*.sh  --backend torchtitan   # default backend=megatron
  └─ launchers/train_torchtitan.py         # mirror of train_megatron.py:
     │                                     #   resolve 6-axis cfg, archive resolved_config.yaml,
     │                                     #   generate <run_dir>/torchtitan.toml, build torchrun cmd
     └─ torchrun -m torchtitan.train --job.config_file <run_dir>/torchtitan.toml  <dotted overrides>
        └─ torchtitan native train loop  (unmodified)
```

The two launchers share `resolve_config` / `archive_resolved_config` /
`_parse_overrides` from `launchers/submit.py`, so config resolution, run-dir
hashing, and launch-metadata recording are identical across backends.

### Backend selection — `--backend` flag

The existing per-optimizer wrappers (`scripts/train_adam.sh`, etc.) gain a
`--backend {megatron,torchtitan}` flag (default `megatron`). The wrapper:

1. injects `backend=<value>` into the Hydra override list (so it lands in the
   resolved config and is hashed), and
2. dispatches to `python -m launchers.train_megatron` or
   `python -m launchers.train_torchtitan` accordingly.

Everything else on the wrapper command line (family, scale, cluster,
experiment, inline overrides) is forwarded unchanged. All three native families
(`llama3`, `qwen3`, `deepseek_v3`) run under `train_adam.sh --backend
torchtitan`; the non-AdamW wrappers reject `--backend torchtitan` with a clear
"not yet supported on torchtitan" error until their optimizers are ported.

### Config → torchtitan mapper — `src/utils/torchtitan_args.py`

The analogue of `src/utils/megatron_args.py`. Input: the resolved slm config.
Output: `(toml_dict, override_list)` written/emitted by the launcher. The model
*architecture* is torchtitan's; the mapper only **selects** the native model and
**registers a flavor** for slm's size, then maps the surrounding knobs:

| slm config | torchtitan |
|---|---|
| `base/family ∈ {llama3, qwen3, deepseek_v3}` | `[model].name = slm_<family>` (an slm-registered clone of torchtitan's native model of that name) |
| `base/scale=*` dims (`num_layers`, `hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_query_groups`, rope base, norm eps) | a registered `slm_<scale>` **flavor** (torchtitan model-args), built by the `custom_import` extension from the resolved config — dims do **not** ride in TOML |
| `base.tokenizer.nominal_vocab_size` (per family) | flavor `vocab_size` (e.g. 128256 for llama3; qwen3/deepseek use their own) |
| `training.global_batch_size`, `model.seq_length`, regime steps, dtype | `[training]` (`global_batch_size`, `seq_len`, `steps`, `mixed_precision_param`) |
| cluster `parallelism.*` | `[parallelism]` (`tensor_parallel_degree`, FSDP `data_parallel_shard_degree`; `expert_parallel_degree` for `deepseek_v3`) |
| `optim.type=adamw`, betas/eps/wd, `training.lr`, WSD schedule | `[optimizer]` (`name="AdamW"`, hyperparams) + `[lr_scheduler]` (`warmup_steps`, `decay_ratio`, `decay_type`, `min_lr_factor`) |
| `cluster.precision` (bf16 first) | `[training].mixed_precision_param` (Float8 deferred) |
| Megatron-only experiment/`patches` knobs torchtitan can't express | **warn-and-skip** (logged once), so the same yaml runs on both backends |

The mapper is a **pure function** (config in, `(toml, overrides)` out) so it is
unit-testable with no GPU, exactly like `megatron_args`.

### The integration surface (what actually takes work)

1. **Same-corpus dataloader — the linchpin.** Register, via torchtitan's
   `experimental.custom_import` hook, a dataloader that wraps Megatron's
   already-vendored `GPTDataset` (`third_party/Megatron-LM/megatron/core/datasets/`)
   so torchtitan reads the **same `.bin/.idx` corpus** the `data=` axis points
   at — no C4, no live tokenization. Byte-identical batches vs the Megatron path
   are a convenient **correctness check** (M2), not a curve-parity requirement.
2. **Scale flavors on native models.** The extension registers an `slm_<family>`
   `TrainSpec` (cloned from torchtitan's native one) whose flavor registry gains
   `slm_<scale>` sized from slm's config. No model code is written; FFN/embeddings
   follow torchtitan's native conventions for that family (a "300m" stays ~300m
   by honoring slm's layer/hidden/ffn dims, but exact-Megatron internals are not
   forced).
3. **Tokenizer / vocab per family.** The corpus is pre-tokenized, so torchtitan's
   tokenizer is bypassed except for `vocab_size`/special ids, taken from the
   family config (llama3 128256; qwen3/deepseek their own). No HF tokenizer
   download on the train path.
4. **WSD scheduler.** Map slm's warmup-stable-decay block to torchtitan's native
   `[lr_scheduler]` (`warmup_steps` + `decay_ratio` + `decay_type` +
   `min_lr_factor`), which already expresses WSD. A custom `LambdaLR` is a tested
   fallback only.
5. **Precision.** bf16 first; Float8 (torchao) is a follow-on toggle from
   `cluster.precision`.

## Reproducibility / run-identity contract

- `backend` is part of the resolved config and prefixes the run name
  (`-torchtitan-`), so torchtitan and Megatron runs get distinct, non-colliding
  run dirs and W&B names.
- Launch metadata + the resolved-config archive record `torchtitan_sha`
  (submodule SHA) next to `megatron_sha`; the generated `torchtitan.toml` is
  archived into the run dir before any GPU work.
- No Megatron-parity numerics gate (exact-structure parity is a non-goal). The
  functional gate is the M3 healthy-curve check below.

## Milestones & acceptance gates

1. **M1 — Skeleton (CI-testable, no GPU).** Submodule + `docs/torchtitan_pin.md`
   + `.gitmodules` entry; `backend` field + `--backend` flag dispatch;
   `launchers/train_torchtitan.py` + `src/utils/torchtitan_args.py`. `--dry-run`
   emits a valid TOML + torchrun command and archives them. Unit tests on the
   mapper.
2. **M2 — Same-corpus dataloader.** The Megatron-`GPTDataset`-backed torchtitan
   dataloader yields correct token batches from the configured `.bin/.idx`;
   byte-identical to the Megatron path on a fixed `(data, seq_len, gbs, seed)` as
   the correctness check.
3. **M3 — Functional training gate (THE acceptance gate).** For each wired family
   (llama3, then qwen3, then deepseek_v3) at a small scale, bf16 + AdamW + WSD: a
   short run **trains and the loss decreases healthily** (no NaNs, sane curve
   shape). An informal side-by-side vs the Megatron run is welcome as a sanity
   signal but gates nothing.
4. **M4 — Follow-ons (separate specs/plans).** Float8/torchao; larger ladder
   rungs + TP/PP; then POET / Muon / nGPT ports onto torchtitan, each its own spec.

## Open risks

- **Config superset gaps.** Some `experiment`/`patches` knobs (sandwich-norm,
  specific MoE router/MLA flags, the `src/patches/` monkey-patches) have no
  torchtitan-native equivalent. Risk is *silent* divergence — mitigation is the
  **warn-and-skip with a one-time log** so the operator sees exactly what the
  torchtitan run ignored. The discovery spike enumerates which knobs map and which
  skip, per family.
- **Scale fidelity, not structure.** A registered `slm_<scale>` flavor must land
  *near* slm's intended non-embedding param count (so a "300m" rung is comparable
  within the torchtitan backend), even though it need not match Megatron's exact
  internals. Sanity-check the param count torchtitan logs at startup.
- **deepseek_v3 surface.** Verified: `DeepSeekV3ModelArgs` is structurally unlike
  the dense families (no `n_kv_heads`; sized via `inter_dim`/`moe_inter_dim`/
  `n_dense_layers`/MLA ranks/`moe_args`), so the shared "clone + override dense
  dims" flavor path does **not** apply. M1 therefore runs deepseek_v3 on a
  **torchtitan-native flavor as-is** (`base.model.titan_flavor`, default
  `debugmodel` for smoke); a proper slm-sized deepseek flavor + the
  `[parallelism].expert_parallel_degree` mapping is a named follow-on, wired last.
- **DCP vs Megatron checkpoints** are not interchangeable — out of scope for
  M1–M3; don't let resume/checkpoint reuse leak into the milestone.
