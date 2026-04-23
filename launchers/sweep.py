"""Ladder sweep orchestration (SPEC.md §9.1, §11.5).

Expands one sweep invocation into N ``submit.py`` calls across scales ×
seeds, with small scales submitted first so failures surface early.

Stubbed until the training loop lands; the skeleton exists so sweep specs
in ``configs/launch/*.yaml`` can be linted and hashed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from launchers.submit import REPO_ROOT, _parse_overrides, submit


def run_sweep(
    base_overrides: list[str],
    *,
    ladder: list[str],
    seeds_per_scale: list[int],
    dry_run: bool,
) -> None:
    if len(ladder) != len(seeds_per_scale):
        raise ValueError("`ladder` and `seeds_per_scale` must have the same length")

    records = []
    # Smallest scales first so early failures kill later submissions.
    for scale, n_seeds in sorted(zip(ladder, seeds_per_scale, strict=True), key=lambda x: x[0]):
        for seed in range(n_seeds):
            overrides = [*base_overrides, f"base/scale={scale}", f"seed={seed}"]
            cfg = _parse_overrides(overrides)
            job_id = submit(cfg, dry_run=dry_run)
            records.append(
                {
                    "scale": scale,
                    "seed": seed,
                    "config_hash": str(cfg._derived.config_hash),
                    "job_id": job_id,
                }
            )

    out = Path(REPO_ROOT) / "runs" / "_last_sweep.json"
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Shared axis overrides, e.g. experiment=optim/muon_hybrid cluster=h800_cn",
    )
    parser.add_argument(
        "--ladder",
        required=True,
        help="Comma-separated scale list, e.g. '600m,1_2b,2_4b'",
    )
    parser.add_argument(
        "--seeds-per-scale",
        required=True,
        help="Comma-separated seed counts aligned with --ladder, e.g. '3,2,1'",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_sweep(
        args.overrides,
        ladder=[s.strip() for s in args.ladder.split(",")],
        seeds_per_scale=[int(s) for s in args.seeds_per_scale.split(",")],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
