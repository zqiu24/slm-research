"""Tests for config_diff.diff_from_champion (SPEC.md §5.3.1)."""

from __future__ import annotations

from src.utils.config_diff import diff_from_champion


def _base() -> dict:
    return {
        "optim": {"type": "adamw", "lr": 1e-3, "betas": [0.9, 0.95]},
        "base": {"family": "qwen3", "scale": "1_2b"},
        "training": {"tokens_per_param": 20},
        "wandb": {"project": "foo"},
        "_derived": {"git_sha": "abc"},
        "parallelism": {"tp": 1, "dp": 8},
    }


def test_identical_returns_champion():
    assert diff_from_champion(_base(), _base()) == "champion"


def test_volatile_fields_ignored():
    cur = _base()
    cur["wandb"]["project"] = "sandbox-alice"
    cur["_derived"]["git_sha"] = "xyz"
    cur["parallelism"] = {"tp": 8, "dp": 1}
    assert diff_from_champion(cur, _base()) == "champion"


def test_single_scalar_change():
    cur = _base()
    cur["optim"]["type"] = "muon_hybrid"
    assert diff_from_champion(cur, _base()) == 'optim.type="muon_hybrid"'


def test_multiple_changes_sorted():
    cur = _base()
    cur["optim"]["type"] = "muon_hybrid"
    cur["optim"]["lr"] = 2e-3
    out = diff_from_champion(cur, _base())
    parts = [p.strip() for p in out.split(",")]
    assert parts == sorted(parts)
    assert 'optim.type="muon_hybrid"' in parts
    assert any(p.startswith("optim.lr=") for p in parts)


def test_new_field_appears():
    cur = _base()
    cur["optim"]["new_knob"] = 7
    assert "optim.new_knob=7" in diff_from_champion(cur, _base())


def test_removed_field_marked():
    cur = _base()
    del cur["optim"]["betas"]
    assert "optim.betas=<removed>" in diff_from_champion(cur, _base())
