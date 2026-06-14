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
    model, _ = _merged_model("nemotron_h", "600m_nemotron_h")
    pattern = str(model["hybrid_layer_pattern"])
    assert len(pattern) == int(model["num_layers"]) == 48
    assert pattern.count("M") == 24
    assert pattern.count("-") == 20
    assert pattern.count("*") == 4
    assert set(pattern) <= {"M", "-", "*"}
