"""train_adam.sh --backend routes to the right launcher and injects backend=."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args):
    # SLM_DRYRUN_PRINT=1 makes the wrapper echo the final `python -m ...` line
    # instead of executing it (added in Step 3), so this test needs no GPU/env.
    env = {"SLM_DRYRUN_PRINT": "1", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        ["bash", str(REPO / "scripts/train_adam.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    return out.stdout + out.stderr


def test_default_backend_is_megatron():
    out = _run(["llama3", "--dry-run"])
    assert "launchers.train_megatron" in out
    assert "backend=torchtitan" not in out


def test_backend_torchtitan_routes_and_injects():
    out = _run(["llama3", "--backend", "torchtitan", "--dry-run"])
    assert "launchers.train_torchtitan" in out
    assert "backend=torchtitan" in out


def test_muon_rejects_torchtitan():
    out = subprocess.run(
        ["bash", str(REPO / "scripts/train_muon.sh"), "llama3", "--backend", "torchtitan"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert out.returncode != 0
    assert "not yet supported on torchtitan" in (out.stdout + out.stderr)
