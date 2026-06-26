"""Smoke tests for optimizer wrapper scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, arch: str):
    return subprocess.run(
        [
            "bash",
            f"scripts/{script}",
            arch,
            "--dry-run",
            "cluster.nodes=1",
            "cluster.gpus_per_node=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_adam_script_supports_llama3():
    proc = _run("train_adam.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "base/family=llama3" not in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "adamw" in proc.stdout


def test_adam_script_supports_minicpm5():
    proc = _run("train_adam.sh", "minicpm5")
    assert "--slm-optimizer" in proc.stdout
    assert "adamw" in proc.stdout
    assert "minicpm5_1b" in proc.stdout


def test_muon_script_supports_deepseek():
    proc = _run("train_muon.sh", "deepseek_v3")
    assert "--optimizer" in proc.stdout
    assert "muon" in proc.stdout
    assert "--multi-latent-attention" in proc.stdout
    # Amendment: confirm we steered to the matching scale (it should show
    # up in the wandb-exp-name and the resolved --num-layers should be 14).
    assert "deepseek_v3_proxy_small" in proc.stdout


def test_muon_script_supports_minicpm5():
    proc = _run("train_muon.sh", "minicpm5")
    assert "--optimizer" in proc.stdout
    assert "muon" in proc.stdout
    assert "minicpm5_1b" in proc.stdout


def test_poet_script_supports_llama3():
    proc = _run("train_poet.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout
    assert "poet" in proc.stdout
    assert "--poet-merge-period" in proc.stdout


def test_muon_kimi_dev_script_routes_to_kimi_optimizer():
    proc = _run("train_muon_dev.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "muon_kimi" in proc.stdout
    assert "slm-zeju-dev" in proc.stdout


def test_poet0_script_supports_llama3():
    proc = _run("train_poet0.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout
    assert "poet" in proc.stdout
    assert "--poet-merge-period" in proc.stdout
    assert "--poet-reinit-period" in proc.stdout


def test_poet_lie_script_supports_llama3():
    proc = _run("train_poet_lie.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout and "poet" in proc.stdout
    assert "--poet-q-optimizer" in proc.stdout
    assert "lie_algebra" in proc.stdout
    assert "--poet-lie-v-mode" in proc.stdout
    assert "--poet-merge-period" in proc.stdout
    assert "--poet-reinit-period" in proc.stdout


def test_poet_lie_alt_script_supports_llama3():
    proc = _run("train_poet_lie_alt.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_algebra" in proc.stdout
    assert "--poet-lie-alternating" in proc.stdout
    assert "--poet-lie-alternate-every" in proc.stdout


def test_poet_lie_rms_script_supports_llama3():
    proc = _run("train_poet_lie_rms.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_algebra" in proc.stdout
    assert "--poet-lie-rms" in proc.stdout
    assert "--poet-lie-rms-c" in proc.stdout


def test_poet_lie_orth_script_supports_llama3():
    proc = _run("train_poet_lie_orth.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_ortho" in proc.stdout
    assert "--poet-lie-ortho-c" in proc.stdout
    assert "--poet-lie-ortho-method" in proc.stdout


def test_poet_lie_orth_update_rms_script_supports_llama3():
    proc = _run("train_poet_lie_orth_update_rms.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_ortho_update_rms" in proc.stdout
    assert "--poet-lie-ortho-update-rms" in proc.stdout
    assert "--poet-lie-ortho-max-angle" in proc.stdout
    assert "--poet-lie-ortho-rms-mode" in proc.stdout


def test_pion_dev_script_routes_to_pion_optimizer():
    proc = _run("train_pion_dev.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "pion" in proc.stdout
    assert "slm-zeju-dev" in proc.stdout
