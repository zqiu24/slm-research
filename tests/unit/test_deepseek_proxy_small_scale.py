"""DeepSeek family + proxy-small scale must satisfy Megatron's layer_freq invariant.

The family YAML sets moe.layer_freq to a 14-element pattern. Megatron asserts
len(layer_freq_eval) == num_layers at startup. Pairing the family with the
right scale is what makes the dry-run-produced command actually launch.
"""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.megatron_args import build_megatron_args


def _eval_layer_freq(expr: str) -> list[int]:
    """Evaluate Megatron's layer_freq mini-language. Restricted to lists / ints."""
    return list(eval(expr, {"__builtins__": {}}, {}))


def test_deepseek_proxy_small_num_layers_matches_family_layer_freq():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_proxy_small",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)

    pattern = _eval_layer_freq(str(cfg.base.model.moe.layer_freq))
    assert len(pattern) == int(cfg.base.model.num_layers), (
        f"layer_freq has {len(pattern)} elements but num_layers is "
        f"{cfg.base.model.num_layers}; Megatron will assert at startup."
    )


def test_deepseek_proxy_small_translator_emits_mla_and_moe_flags():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_proxy_small",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)

    args = build_megatron_args(cfg)
    assert "--multi-latent-attention" in args
    assert "--num-experts" in args
    i = args.index("--num-layers")
    assert args[i + 1] == "14"
