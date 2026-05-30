# torchtitan pin and bump history

torchtitan is included under `third_party/torchtitan` as a submodule. Like
Megatron-LM, the pin is conservative: torchtitan's parallelize/quantization
APIs move fast and silently change numerics.

## Current pin

- Tag / SHA: `v0.2.2` -> `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`
- Pinned on: 2026-05-30
- Reason: first import. `v0.2.2` is the newest release tag (all torchtitan
  releases are GitHub pre-releases; we treat the newest tag as "last stable"
  per the pin-to-a-tag contract used for Megatron-LM, SPEC.md §4.1).

To initialise on a fresh clone:

    git submodule update --init --recursive

## Bump procedure

1. Branch `torchtitan-bump-<date>`.
2. `cd third_party/torchtitan && git fetch --tags && git checkout <new-tag>`.
3. Re-run the discovery checks in `docs/torchtitan_api_notes.md` — if any
   recorded symbol (train-spec registration, JobConfig key, dataloader
   component API) moved, update `src/titan_ext/` and the notes.
4. `pytest tests/unit -k torchtitan` and `pytest tests/integration/test_titan_megatron_data_parity.py`.
5. Rerun the per-family training-health gate (Task 12); curves must stay healthy.
6. Update this doc (SHA, date, reason, any API changes).

## Bump log

| Date | New SHA | Prior SHA | Reason | API changes |
|---|---|---|---|---|
| 2026-05-30 | `v0.2.2` (`73a0e6979`) | — | first import | n/a |
