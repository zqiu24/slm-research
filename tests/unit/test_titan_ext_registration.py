from __future__ import annotations

import dataclasses
import importlib

import pytest
from omegaconf import OmegaConf

from launchers.submit import _parse_overrides, resolve_config


def test_build_slm_flavor_overrides_only_template_fields():
    """build_slm_flavor clones a native template and overrides only the dim
    fields the template actually has — so it works across the different
    per-family model-args classes (llama3 vs qwen3 vs deepseek_v3) and silently
    ignores slm fields torchtitan doesn't model (e.g. ffn_hidden_size)."""
    from src.titan_ext.model_flavor import build_slm_flavor

    @dataclasses.dataclass
    class FakeArgs:  # stands in for a native TransformerModelArgs
        dim: int = 4096
        n_layers: int = 32
        n_kv_heads: int = 8
        vocab_size: int = 1000
        # NOTE: no ffn_hidden_size field on purpose

    out = build_slm_flavor(
        FakeArgs(),
        {
            "dim": 1024,
            "n_layers": 12,
            "n_kv_heads": 4,
            "vocab_size": 128256,
            "ffn_hidden_size": 2560,
        },
    )
    assert out.dim == 1024 and out.n_layers == 12 and out.n_kv_heads == 4
    assert out.vocab_size == 128256  # ffn_hidden_size was ignored, not a crash


def test_import_with_env_registers_slm_family_spec(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")  # this test needs the real registry
    cfg = _parse_overrides(
        ["base/family=llama3", "base/scale=300m", "experiment=optim/adam", "backend=torchtitan"]
    )
    resolve_config(cfg)
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    monkeypatch.setenv("SLM_RESOLVED_CONFIG", str(resolved))

    import src.titan_ext as ext

    importlib.reload(ext)  # re-run registration with the env in place

    from torchtitan.protocols.train_spec import get_train_spec

    spec = get_train_spec("slm_llama3")
    assert "slm_300m" in spec.model_args  # our flavor is in the registry
