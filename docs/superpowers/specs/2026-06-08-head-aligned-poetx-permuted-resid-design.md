# head-aligned → POETX (permuted multi-block residual) + head × alternating — Design Spec

**Date:** 2026-06-08
**Status:** approved (brainstorm), pending implementation plan
**Depends on:** the integrated alternating-POETX work (`2026-06-08-alternating-poetx-integrated-design.md`) — POETXLinear must already carry the `alternating` mode. Order: that plan first, then this one.

## Goal

Port head-aligned attention rotation onto the POETX forward-frame layer as a thin `POETXLinear` subclass, and — the point — give the **residual side a real POETX permutation + multiple blocks** (the "general" head-aligned design the current layer cannot express). Then test whether this corrected head-aligned, combined with the alternating champion, **flips the −0.016 head penalty**.

## Why

- `HeadAlignedPOETLinear` today subclasses the **old** `POETLinear` and hard-codes a **single dense, perm-free residual side**. That is correct *only* at `resid_block_count=1` (where Ψ conjugating one dense block is a redundant relabeling). In general the residual side needs its permutation: POET's expressivity comes from cheap **block-diagonal** rotations + a **resampled permutation** that reshuffles which neurons share a block — and Ψ only does work when there are ≥2 blocks.
- The current layer therefore can't even represent a permuted multi-block residual. Porting head onto `POETXLinear` (which carries real `perm_in`/`perm_out`) restores it naturally.
- **Hypothesis:** the perm-free single-block residual under-parameterizes head-aligned and is part of why it hurts (−0.014 ortho, more in RMS). A permuted multi-block residual + the alternating champion may make per-head rotation finally competitive.

## Background facts (verified)

- `HeadAlignedPOETLinear` has both `oft_R_in`/`oft_R_out`, is built by the walk with `resid_block_count = block_count` (= 1 in deployed configs), and `head_side="out"` for q/k/v, `"in"` for o.
- `POETXSingleStepFunction` is already **perm-aware** (uses `perm_in_inv`/`perm_out_inv` in the backward conj) and supports **decoupled block sizes** (`block_size_in ≠ block_size_out`). So an asymmetric-perm, asymmetric-block POETX layer needs **no new forward function**.
- `lie_alternating` + `head_aligned_attn` is not blocked (only the research flag `single_step_x_alternating` is). The optimizer's `alternating` mode is layer-agnostic.

## Design

### Phase 1 — `HeadAlignedPOETXLinear(POETXLinear)`

A thin subclass; all compute (forward / backward / merge) is **inherited** from `POETXLinear`. The subclass only configures the perms and block sizes asymmetrically at construction:

- **Head side** (`out` for q/k/v, `in` for o): perm = **identity**, block size = `head_dim` (block-diagonal per head, no cross-head mixing).
- **Residual side** (the `hidden_size` side): perm = **random** (POETX default), block count = `head_resid_block_count` (**> 1**), block size = `hidden_size / head_resid_block_count`.
- Forward = inherited bare-GEMM on the forward-frame weight; backward conj applies identity on the head side (no-op) and the random perm on the residual side; merge = inherited POETX round-trip fold.
- **Inherits `POETXLinear`'s `alternating` mode for free** (from the prerequisite plan), so head + alternating composes with no extra layer code.
- Walk builds it for attention q/k/v/o when `head_aligned_attn` + `single_step_x`.
- **New knob:** `head_resid_block_count` (config/CLI), default a natural value such as `num_heads` (→ residual blocks are also `head_dim`-sized); swept in Phase 2.

**Not bit-identical to the old `HeadAlignedPOETLinear`** — the residual perm/blocks differ by design. So the parity anchor is the **POET reference chain** (`R_out · W · R_inᵀ` with the asymmetric perms/blocks), not the old layer.

### Phase 2 — Experiment (head × alternating, POETX-native)

| | both-sides | + alternating |
|---|---|---|
| head-off | 3.5504 (have) | **3.5332 champion (have)** |
| head-on (POETX, permuted multi-block resid) | optional | **new run(s) — sweep `head_resid_block_count`** |

- Primary: does head-on (permuted multi-block resid) **+ alternating** beat the head-off champion (3.5332)? If yes, per-head rotation flips to helping once the residual side is properly permuted.
- Sweep `head_resid_block_count` (e.g. {num_heads, num_heads/2, …}); optional `alternate_every=2`.
- Record in POET_dev.md (§2.1 head row, §2.3, §2.5, §2.6).

## Components & files

| File | Change |
|---|---|
| `third_party/poet_torch/head_aligned_poetx_layer.py` (new) | `HeadAlignedPOETXLinear(POETXLinear)` — asymmetric perms (head=identity, resid=random) + asymmetric blocks |
| `third_party/poet_torch/__init__.py` | export |
| `src/optim/poet_layers.py` | walk builds `HeadAlignedPOETXLinear` for attention under `head_aligned_attn` + `single_step_x`; thread `head_resid_block_count` |
| `src/utils/megatron_args.py`, `launchers/pretrain_gpt_slm.py`, patches | `head_resid_block_count` flag + validation (requires `head_aligned_attn` + `single_step_x`); allow `head_aligned_attn` + `single_step_x` |
| `configs/experiments/optim/poet_lie_orth_head_alt.yaml` (new), doc, launcher | head-on POETX + alternating experiment |
| tests | structure; forward-chain parity; walk-selection; full CPU suite |

## Testing & correctness gates

- **Structure (CPU):** head side is block-diagonal per head (identity perm, `head_dim` blocks); residual side is permuted with `head_resid_block_count` blocks.
- **Forward-chain parity (CPU):** `HeadAlignedPOETXLinear` forward == the POET reference chain `R_out · W · R_inᵀ` with the asymmetric perms/blocks, fp64.
- **Merge (CPU):** fold is spectrum-preserving / round-trips like POETXLinear; alternating active-only fold inherited correctly.
- **Walk-selection (CPU):** attention q/k/v/o → `HeadAlignedPOETXLinear` with the right `head_side`/blocks; non-attention → `POETXLinear`.
- **Full CPU suite green.**
- **GPU (user):** Phase 2 head × alternating sweep vs the 3.5332 champion.

## Out of scope

- Bit-identical reproduction of the old perm-free single-block head layer (intentionally superseded; the old layer stays for legacy runs).
- Re-tuning head lr/c or larger-scale head tests (separate hypotheses, not picked).

## Open questions

- Default / sweep range for `head_resid_block_count` — start at `num_heads` (residual blocks = `head_dim`), confirm in plan.
