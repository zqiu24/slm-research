"""Resolve slm config and launch Megatron through the slm per-rank entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from launchers.submit import REPO_ROOT, _parse_overrides, archive_resolved_config, resolve_config
from src.utils.megatron_args import build_megatron_args


def _launch_nnodes() -> int:
    """Number of nodes joining *this* torchrun rendezvous.

    Read from the launch environment, not ``cfg.cluster.nodes``: the latter is
    the experiment's intended multi-node scale (used for parallelism math),
    while the rendezvous size is set by whatever launches the job (e.g. SLURM).
    Defaults to a single node so local runs don't block forever in rendezvous
    waiting for peers that never join.
    """
    for var in ("NNODES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        value = os.environ.get(var)
        if value:
            return int(value)
    return 1


_ENTRYPOINT_MODULES = {
    "gpt": "launchers.pretrain_gpt_slm",
    "mamba": "launchers.pretrain_mamba_slm",
}


def build_torchrun_command(cfg) -> list[str]:
    entrypoint = str(cfg.base.model.get("entrypoint", "gpt"))
    if entrypoint not in _ENTRYPOINT_MODULES:
        raise ValueError(
            f"Unknown base.model.entrypoint {entrypoint!r}; "
            f"expected one of {sorted(_ENTRYPOINT_MODULES)}"
        )
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
        _ENTRYPOINT_MODULES[entrypoint],
        "--slm-config-path",
        os.fspath(Path(REPO_ROOT) / cfg._derived.run_dir / "resolved_config.yaml"),
    ]
    cmd.extend(build_megatron_args(cfg))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args, leftover = parser.parse_known_args()
    overrides = list(args.overrides) + list(leftover)

    cfg = _parse_overrides(overrides)
    resolve_config(cfg)
    archive = archive_resolved_config(cfg)
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
            os.fspath(Path(REPO_ROOT) / "third_party" / "Megatron-LM"),
            env.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
