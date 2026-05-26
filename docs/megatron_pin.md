# Megatron-LM pin and bump history

Megatron-LM is included under `third_party/Megatron-LM` as a submodule. The
pin is intentionally conservative — TE + mcore API surface changes silently
break FP8 numerics and ModuleSpec layouts.

## Current pin

- Tag / SHA: `core_v0.17.0` → `9539a12e1b04a68423f57b3eb41d6125161dca24`
- Pinned on: 2026-05-13
- Reason: NVIDIA resumed cutting annotated release tags with `core_v0.15.0`
  (2025-12-17), so we're back on the "pin to a tag" contract. `core_v0.17.0`
  is the latest stable release tag (commit dated 2026-04-14, identical SHA
  to `core_v0.17.0rc0` — the rc was promoted to release without changes).
  The `core_r0.17.0` release branch has 24 backport commits past the tag as
  of 2026-05-11 (async ckpt fixes, SafeUnpickler, `transformers<=5.3.0`
  cap relaxation, hybrid-EP permute fusion); none are needed by the current
  pretraining path, so we stay on the tagged SHA rather than the moving
  branch tip.

To initialise the submodule on a fresh clone:

```bash
git clone --recurse-submodules https://github.com/zqiu24/slm-research.git
# or, if already cloned:
git submodule update --init --recursive
```

## Bump procedure (SPEC.md §4.1)

1. Create a branch `megatron-bump-<date>`.
2. `cd third_party/Megatron-LM && git fetch && git checkout <new-sha>`.
3. In the repo root, re-run `apply_patches(<all>)`. Inspect each patch in
   `src/patches/` — if the upstream function moved or changed signature, fix
   or retire the patch.
4. `pytest tests/numerics/` (patch-neutrality must still hold).
5. Rerun the champion config at 1.2B and 2.4B. Loss curves must match the
   prior champion within seed-variance bounds.
6. Update this doc (SHA, date, reason for bump, any patch changes).
7. Merge to `main`.

## Bump log

| Date | New SHA | Prior SHA | Reason | Patch changes |
|---|---|---|---|---|
| 2026-04-23 | `core_r0.16.0` tip (`ddc0d6774`) | — | first import | none |
| 2026-05-13 | `core_v0.17.0` (`9539a12e1`) | `ddc0d6774` | move to first tagged 0.17 release; align with newly-resumed annotated-tag cadence | none — `src/patches/` not yet populated; numerics gate deferred until `tests/numerics/` lands (SPEC.md §12) |
