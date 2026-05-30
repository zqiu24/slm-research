"""Resolve slm config and launch torchtitan through its native train entrypoint.

`python -m launchers.train_torchtitan <hydra overrides> [--dry-run]`

Mirror of launchers.train_megatron: resolve the same 6-axis config, archive
resolved_config.yaml, emit <run_dir>/torchtitan.toml, and build the
`torchrun -m torchtitan.train` command. The slm wiring (slm_<family> TrainSpec,
slm_<scale> flavor, Megatron-indexed dataloader, WSD scheduler) is injected via
torchtitan's experimental.custom_import hook — the vendored submodule is never
edited.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover - dep installed on compute nodes
    tomli_w = None

from launchers.submit import (
    REPO_ROOT,
    _parse_overrides,
    archive_resolved_config,
    resolve_config,
)
from launchers.train_megatron import _launch_nnodes
from src.utils.torchtitan_args import build_torchtitan_config, unmapped_megatron_knobs

TORCHTITAN_ROOT = Path(REPO_ROOT) / "third_party" / "torchtitan"
MEGATRON_ROOT = Path(REPO_ROOT) / "third_party" / "Megatron-LM"

# VERIFIED against v0.2.2: train.py runs importlib.import_module(
#   job_config.experimental.custom_import) before get_train_spec lookup, so the
# CLI form below imports src.titan_ext (which registers the slm_<family> spec).
CUSTOM_IMPORT_FLAG = "--experimental.custom_import"


def _write_toml(cfg) -> Path:
    toml_dict, _ = build_torchtitan_config(cfg)
    run_dir = Path(REPO_ROOT) / cfg._derived.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "torchtitan.toml"
    if tomli_w is None:
        raise RuntimeError("tomli-w is required to emit torchtitan.toml; pip install tomli-w")
    with out.open("wb") as fh:
        tomli_w.dump(toml_dict, fh)
    return out


def build_torchrun_command(cfg) -> list[str]:
    toml_path = Path(REPO_ROOT) / cfg._derived.run_dir / "torchtitan.toml"
    _, overrides = build_torchtitan_config(cfg)
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(cfg.cluster.gpus_per_node),
        "--nnodes",
        str(_launch_nnodes()),
        "--node_rank",
        str(os.environ.get("NODE_RANK", "0")),
        "--master_addr",
        str(os.environ.get("MASTER_ADDR", "localhost")),
        "--master_port",
        str(os.environ.get("MASTER_PORT", "6000")),
        "-m",
        "torchtitan.train",
        "--job.config_file",
        os.fspath(toml_path),
        CUSTOM_IMPORT_FLAG,
        "src.titan_ext",
    ]
    cmd.extend(overrides)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args, leftover = parser.parse_known_args()
    overrides = list(args.overrides) + list(leftover)

    cfg = _parse_overrides(overrides)
    if "backend" not in cfg:
        cfg.backend = "torchtitan"
    resolve_config(cfg)
    for note in unmapped_megatron_knobs(cfg):  # warn-and-skip Megatron-only knobs
        print(f"[torchtitan] skipping {note}")
    archive = archive_resolved_config(cfg)
    _write_toml(cfg)
    cmd = build_torchrun_command(cfg)

    payload = {
        "run_name": str(cfg._derived.run_name),
        "archive": os.fspath(archive),
        "command": cmd,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.fspath(REPO_ROOT),
            os.fspath(TORCHTITAN_ROOT),
            os.fspath(MEGATRON_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    # src.titan_ext (imported via experimental.custom_import inside torchtitan)
    # reads this to build the slm_<scale> flavor dims that can't ride in TOML.
    env["SLM_RESOLVED_CONFIG"] = os.fspath(
        Path(REPO_ROOT) / cfg._derived.run_dir / "resolved_config.yaml"
    )
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
