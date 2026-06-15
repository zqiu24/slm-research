"""CI gate: every bake-off scale file realizes its declared
base.non_embedding_params budget within ±2% (computed by arch_params)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.arch_params import active_non_embedding_params, non_embedding_params

REPO_ROOT = Path(__file__).resolve().parents[2]

BAKEOFF_PAIRS = [
    ("deepseek_v3", "600m_deepseek_v3"),
    ("deepseek_v3_dense", "600m_deepseek_v3_dense"),
    ("qwen3_next", "600m_qwen3_next"),
    ("nemotron_h", "600m_nemotron_h"),
    ("gemma3", "600m_gemma3"),
    # dense-architecture bake-off: plain dense GQA transformers, param-matched
    ("qwen3", "600m_qwen3"),
    ("llama3", "600m_llama3"),
    ("minicpm", "600m_minicpm"),
    # dense-architecture bake-off, 1B non-embedding budget
    ("qwen3", "1b_qwen3"),
    ("llama3", "1b_llama3"),
    ("minicpm", "1b_minicpm"),
    ("gemma3", "1b_gemma3"),
    ("deepseek_v3_dense", "1b_deepseek_v3_dense"),
    ("nemotron_h", "1b_nemotron_h"),
]


def _merged_model(family: str, scale: str):
    fam = OmegaConf.load(REPO_ROOT / f"configs/base/family/{family}.yaml")
    sc = OmegaConf.load(REPO_ROOT / f"configs/base/scale/{scale}.yaml")
    merged = OmegaConf.merge(fam, sc)
    model = OmegaConf.to_container(merged.base.model, resolve=True)
    return model, int(merged.base.non_embedding_params)


@pytest.mark.parametrize("family,scale", BAKEOFF_PAIRS)
def test_bakeoff_scale_within_budget(family, scale):
    model, budget = _merged_model(family, scale)
    actual = non_embedding_params(model)
    rel = (actual - budget) / budget
    assert abs(rel) <= 0.02, f"{family}/{scale}: {actual:,} vs {budget:,} ({rel:+.2%})"


@pytest.mark.parametrize("family,scale", BAKEOFF_PAIRS)
def test_active_not_above_total(family, scale):
    model, _ = _merged_model(family, scale)
    assert active_non_embedding_params(model) <= non_embedding_params(model)


def test_deepseek_v3_dense_is_dense():
    # Dense DeepSeek = MLA kept, MoE off -> every layer dense, active == total.
    model, _ = _merged_model("deepseek_v3_dense", "600m_deepseek_v3_dense")
    assert model.get("multi_latent_attention") is True
    assert not (model.get("moe") or {}).get("enabled", False)
    assert active_non_embedding_params(model) == non_embedding_params(model)


def test_nemotron_pattern_shape():
    # 600m: (A+B)*2 = 48 layers, even Mamba/FFN split (§2.1) -> 22 M / 22 - / 4 *.
    model, _ = _merged_model("nemotron_h", "600m_nemotron_h")
    pattern = str(model["hybrid_layer_pattern"])
    assert len(pattern) == int(model["num_layers"]) == 48
    assert pattern.count("M") == 22
    assert pattern.count("-") == 22
    assert pattern.count("*") == 4
    assert set(pattern) <= {"M", "-", "*"}


def test_nemotron_1b_pattern_shape():
    # 1B: same (A+B) unit tiled 3x = 72 layers, even split -> 33 M / 33 - / 6 *.
    model, _ = _merged_model("nemotron_h", "1b_nemotron_h")
    pattern = str(model["hybrid_layer_pattern"])
    assert len(pattern) == int(model["num_layers"]) == 72
    assert pattern.count("M") == 33
    assert pattern.count("-") == 33
    assert pattern.count("*") == 6
    assert set(pattern) <= {"M", "-", "*"}
    assert pattern[0] == "M" and pattern[-1] == "-"  # §2.1: first=Mamba, last=FFN
