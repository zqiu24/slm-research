"""Tests for slm-research config-axis composition."""

from __future__ import annotations

from pathlib import Path

import yaml

from launchers.submit import _parse_overrides, resolve_config


def test_parse_overrides_loads_defaults_and_data_axis():
    cfg = _parse_overrides([])

    assert cfg.base.family == "qwen3"
    assert cfg.base.scale == "1_2b"
    assert cfg.experiment.name == "adam"
    assert cfg.cluster.name == "h800_cn"
    assert cfg.data.name == "nemotron_cc_v2_llama31_8b"
    assert cfg.data.path == (
        "/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/"
        "nemotron_cc_v2_high_quality_text_document_llama31_8b"
    )


def test_parse_overrides_loads_nested_experiment_value():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet",
            "training_regime=ablation_40x",
            "cluster=h100_de",
            "data=nemotron_cc_v2_scratch_qwen3",
            "seed=7",
        ]
    )

    assert cfg.base.family == "llama3"
    assert cfg.base.scale == "600m"
    assert cfg.experiment.name == "poet"
    assert cfg.training.tokens_per_param == 40
    assert cfg.cluster.name == "h100_de"
    assert cfg.data.name == "nemotron_cc_v2_scratch_qwen3"
    assert cfg.seed == 7


def test_data_catalog_prefixes_point_at_bin_and_idx_files():
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "configs/data").glob("*.yaml")):
        data = yaml.safe_load(path.read_text())["data"]
        prefix = Path(data["path"])
        assert not str(prefix).endswith(".bin")
        assert not str(prefix).endswith(".idx")
        assert Path(str(prefix) + ".bin").exists(), f"{path} points at missing bin"
        assert Path(str(prefix) + ".idx").exists(), f"{path} points at missing idx"


def test_fixed_regime_total_tokens_is_scale_independent():
    def resolved(scale: str):
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                f"base/scale={scale}",
                "experiment=champion",
                "training_regime=fixed_1b",
                "cluster=h800_cn",
            ]
        )
        resolve_config(cfg)
        return cfg

    small, large = resolved("60m"), resolved("300m")
    assert int(small.training.total_tokens) == 1_000_000_000
    assert int(large.training.total_tokens) == 1_000_000_000
    assert small.training.tokens_per_param is None


def test_all_fixed_regimes_parse_to_expected_budgets():
    expected = {
        "fixed_500m": 500_000_000,
        "fixed_1b": 1_000_000_000,
        "fixed_10b": 10_000_000_000,
        "fixed_50b": 50_000_000_000,
        "fixed_100b": 100_000_000_000,
    }
    for regime, budget in expected.items():
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                "base/scale=60m",
                "experiment=champion",
                f"training_regime={regime}",
                "cluster=h800_cn",
            ]
        )
        resolve_config(cfg)
        assert int(cfg.training.total_tokens) == budget, regime
