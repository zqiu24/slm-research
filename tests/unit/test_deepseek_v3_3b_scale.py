"""DeepSeek-V3 3B scale parity with Megatron-poet's DeepSeek-3B.yaml.

Mirrors the dim / MoE checks from test_deepseek_proxy_small_scale.py for the
new deepseek_v3_3b scale, and verifies the group-routing flags are dropped
when the scale sets router_group_topk / router_num_groups to null.
"""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.megatron_args import build_megatron_args


def _eval_layer_freq(expr: str) -> list[int]:
    return list(eval(expr, {"__builtins__": {}}, {}))


def _args_to_map(args: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def test_deepseek_v3_3b_layer_freq_matches_num_layers():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_3b",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)

    pattern = _eval_layer_freq(str(cfg.base.model.moe.layer_freq))
    assert len(pattern) == int(cfg.base.model.num_layers) == 12


def test_deepseek_v3_3b_emits_mla_and_moe_dims_from_recipe():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_3b",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)
    amap = _args_to_map(build_megatron_args(cfg))

    assert amap["--num-layers"] == "12"
    assert amap["--hidden-size"] == "1280"
    assert amap["--ffn-hidden-size"] == "7168"
    assert amap["--num-attention-heads"] == "16"
    assert amap["--kv-channels"] == "128"
    assert amap["--multi-latent-attention"] is True
    assert amap["--q-lora-rank"] == "1536"
    assert amap["--kv-lora-rank"] == "512"
    assert amap["--v-head-dim"] == "128"
    assert amap["--num-experts"] == "64"
    assert amap["--moe-router-topk"] == "6"
    assert amap["--moe-ffn-hidden-size"] == "896"
    assert amap["--moe-shared-expert-intermediate-size"] == "1792"
    assert amap["--moe-token-dispatcher-type"] == "alltoall"
    assert amap["--mtp-num-layers"] == "1"
    assert amap["--mtp-loss-scaling-factor"] == "0.3"


def test_deepseek_v3_3b_drops_group_routing_flags():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=deepseek_v3_3b",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)
    args = build_megatron_args(cfg)

    assert "--moe-router-group-topk" not in args
    assert "--moe-router-num-groups" not in args
    # DeepEP and pre-softmax are disabled by the scale.
    assert "--moe-enable-deepep" not in args
    assert "--moe-router-pre-softmax" not in args


def test_deepseek_v3_proxy_small_still_emits_group_routing_flags():
    # Regression guard: existing scale must continue to emit n-group routing.
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
    amap = _args_to_map(build_megatron_args(cfg))

    assert amap["--moe-router-group-topk"] == "4"
    assert amap["--moe-router-num-groups"] == "8"
