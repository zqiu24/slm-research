# Megatron-LM pin and bump history

Megatron-LM is included under `third_party/Megatron-LM` as a submodule. The
pin is intentionally conservative — TE + mcore API surface changes silently
break FP8 numerics and ModuleSpec layouts.

## Current pin

- Branch / SHA: `core_r0.16.0` tip → `ddc0d6774783b032ddceacc5714e653651daecb9`
- Pinned on: 2026-04-23
- Reason: initial import. Upstream stopped cutting release *tags* after
  `core_r0.11.0` (Feb 2025); stable versions now live on release *branches*.
  `core_r0.16.0` is the branch NVIDIA is currently backporting fixes to
  (last commit 2026-04-17). `core_r0.17.0` was branched three days before
  this pin (2026-04-20) and is still churning, so we stay one minor back
  for stability. We pin to the specific SHA, not the moving branch ref,
  to preserve reproducibility.

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
