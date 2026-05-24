"""Unit tests for translating slm configs into Megatron CLI args."""

from __future__ import annotations

from omegaconf import OmegaConf

from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args


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


def test_llama3_adam_args_include_dense_gqa_rope_and_data_prefix():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--use-mcore-models"] is True
    assert args["--num-layers"] == "40"
    assert args["--hidden-size"] == "1280"
    assert args["--group-query-attention"] is True
    assert args["--num-query-groups"] == "4"
    assert args["--position-embedding-type"] == "rope"
    assert args["--rotary-base"] == "500000"
    assert args["--tokenizer-type"] == "HuggingFaceTokenizer"
    assert args["--data-path"].endswith("nemotron_cc_v2_high_quality_text_document_llama31_8b")
    assert args["--optimizer"] == "adam"


def test_deepseek_args_include_mla_moe_and_deepseek_router_knobs():
    cfg = _parse_overrides(
        [
            "base/family=deepseek_v3",
            "base/scale=600m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--multi-latent-attention"] is True
    assert args["--q-lora-rank"] == "1536"
    assert args["--kv-lora-rank"] == "512"
    assert args["--qk-head-dim"] == "128"
    assert args["--qk-pos-emb-head-dim"] == "64"
    assert args["--v-head-dim"] == "128"
    assert args["--num-experts"] == "64"
    assert args["--moe-router-topk"] == "8"
    assert args["--moe-router-score-function"] == "sigmoid"
    assert args["--moe-router-enable-expert-bias"] is True
    assert args["--enable-experimental"] is True


def test_muon_args_use_megatron_muon_and_disable_dist_optimizer_overlap():
    cfg = _parse_overrides(["experiment=optim/muon_hybrid"])
    args = build_megatron_args(cfg)
    amap = _args_to_map(args)

    assert amap["--optimizer"] == "muon"
    assert "--use-distributed-optimizer" not in args
    assert "--overlap-grad-reduce" not in args
    assert "--overlap-param-gather" not in args
    assert amap["--muon-num-ns-steps"] == "5"
    assert amap["--muon-momentum"] == "0.95"


def test_poet_args_use_slm_optimizer_and_keep_megatron_optimizer_adam():
    cfg = _parse_overrides(["experiment=optim/poet"])
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--optimizer"] == "adam"
    assert args["--slm-optimizer"] == "poet"
    assert args["--poet"] is True
    assert args["--poet-block-size"] == "256"
    assert args["--poet-merge-period"] == "200"


def test_poet_argv_includes_cache_mode():
    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "block_size": 256,
                    "cache_mode": "cached_fwd_bwd",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 200,
                    "scale": 1.0,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert "--poet-cache-mode" in args
    assert "cached_fwd_bwd" in args
