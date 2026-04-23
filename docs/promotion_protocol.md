# Promotion protocol

Authoritative source: SPEC.md §11. This doc is the short version.

## Branching

- `main` carries the champion. Merges happen only via promotion events.
- Each researcher has a long-lived feature branch (e.g. `alice/muon`).
- Feature branches rebase on `main` at promotion events.

## Weekly cycle

Each person submits their current best candidate to the weekly 2.4B
promotion gate. One person (rotating) reviews outputs, flags regressions,
and verifies W&B tag hygiene.

## Monthly cycle

1. `python -m tools.gen_monthly_report month=YYYY-MM` builds the report.
2. Team reviews; winner is chosen by primary-metric improvement at 2.4B,
   corroborated by ≥2σ agreement across seeds **and** a favorable
   600M→1.2B→2.4B slope.
3. Winner's PR is code-reviewed by ≥1 teammate.
4. Merge to `main`; new champion baseline runs start at 1.2B and 2.4B.
5. 7B HPC run scheduled for next month's anchor.

## Promotion PR template

- Config diff vs the current champion.
- W&B report link showing slope and 2.4B comparison.
- Smoke-run CI passing.
- `config_hash` before and after promotion.
- One-paragraph description of what the method does and why it helps.

## End-of-month audit

`tools/validate_ladder.py` runs automatically and flags runs with missing
tags, missing patch hashes, wrong dataset hash, dirty git SHA in shared
projects, or lone `config_hash` (no seed replication). Flagged runs are
tagged `status:deprecated` and excluded from the report.
