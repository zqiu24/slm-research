# Megatron-Bridge pin and bump history

Megatron-Bridge is included under `third_party/Megatron-Bridge` as a
submodule. Same conservative pin contract as Megatron-LM and torchtitan:
the bridge's checkpoint/conversion APIs track mcore and HF releases closely
and move fast between tags.

## Current pin

- Tag / SHA: `v0.4.2` → `c810129341a84e58f4cbed3093f70668a088c028`
- Pinned on: 2026-06-12
- Reason: first import. `v0.4.2` is the latest stable GitHub release
  (published 2026-05-28, not a pre-release). The newer `26.04-alpha.rc1/rc2`
  tags are alpha pre-releases and are skipped per the pin-to-a-tag contract
  (SPEC.md §4.1).

To initialise on a fresh clone:

    git submodule update --init --recursive

## Bump procedure

1. Branch `megatron-bridge-bump-<date>`.
2. `cd third_party/Megatron-Bridge && git fetch --tags && git checkout <new-tag>`.
3. Check compatibility against the current Megatron-LM pin
   (`docs/megatron_pin.md`) — the bridge tracks mcore API surface, so bump
   both together when the bridge requires a newer mcore.
4. Re-run any tests that exercise the bridge conversion path.
5. Update this doc (SHA, date, reason, any API changes).

## Bump log

| Date | New SHA | Prior SHA | Reason | API changes |
|---|---|---|---|---|
| 2026-06-12 | `v0.4.2` (`c81012934`) | — | first import | n/a |
