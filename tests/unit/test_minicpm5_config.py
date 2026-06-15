"""MiniCPM5 config and Megatron argv checks."""

from __future__ import annotations

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


def test_minicpm5_1b_config_matches_hf_shape():
    cfg = _parse_overrides(
        [
            "base/family=minicpm5",
            "base/scale=minicpm5_1b",
            "experiment=optim/adam",
        ]
    )

    assert cfg.base.family == "minicpm5"
    assert cfg.base.family_version == "5.0"
    assert cfg.base.non_embedding_params == 679_552_512
    assert cfg.base.model.num_layers == 24
    assert cfg.base.model.hidden_size == 1536
    assert cfg.base.model.ffn_hidden_size == 4608
    assert cfg.base.model.num_attention_heads == 16
    assert cfg.base.model.num_query_groups == 2
    assert cfg.base.model.head_dim == 128
    assert cfg.base.model.max_position_embeddings == 131072
    assert cfg.base.model.tie_embeddings is False


def test_minicpm5_1b_megatron_args_include_gqa_rope_and_untied_embeddings():
    cfg = _parse_overrides(
        [
            "base/family=minicpm5",
            "base/scale=minicpm5_1b",
            "experiment=optim/adam",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--num-layers"] == "24"
    assert args["--hidden-size"] == "1536"
    assert args["--ffn-hidden-size"] == "4608"
    assert args["--num-attention-heads"] == "16"
    assert args["--group-query-attention"] is True
    assert args["--num-query-groups"] == "2"
    assert args["--kv-channels"] == "128"
    assert args["--seq-length"] == "4096"
    assert args["--max-position-embeddings"] == "131072"
    assert args["--rotary-base"] == "5000000"
    assert args["--norm-epsilon"] == "1e-06"
    assert args["--untie-embeddings-and-output-weights"] is True
    assert "--qk-layernorm" not in args
    assert "--multi-latent-attention" not in args


def test_minicpm5_600m_is_depth_scaled_official_1b():
    """600M variant: identical per-layer config to the official 1B, 21 layers."""
    cfg = _parse_overrides(
        [
            "base/family=minicpm5",
            "base/scale=minicpm5_600m",
            "experiment=optim/adam",
        ]
    )

    assert cfg.base.family == "minicpm5"
    assert cfg.base.scale == "minicpm5_600m"
    assert cfg.base.non_embedding_params == 594_608_640
    # per-layer config matches minicpm5_1b exactly; only depth differs (24 -> 21)
    assert cfg.base.model.num_layers == 21
    assert cfg.base.model.hidden_size == 1536
    assert cfg.base.model.ffn_hidden_size == 4608
    assert cfg.base.model.num_attention_heads == 16
    assert cfg.base.model.num_query_groups == 2
    assert cfg.base.model.head_dim == 128
    assert cfg.base.model.tie_embeddings is False

    args = _args_to_map(build_megatron_args(cfg))
    assert args["--num-layers"] == "21"
    assert args["--hidden-size"] == "1536"
    assert args["--num-query-groups"] == "2"
    assert args["--kv-channels"] == "128"
    assert args["--rotary-base"] == "5000000"
    assert args["--untie-embeddings-and-output-weights"] is True


def test_mock_data_override_emits_megatron_mock_data_flag():
    cfg = _parse_overrides(
        [
            "base/family=minicpm5",
            "base/scale=minicpm5_1b",
            "experiment=optim/adam",
            "data.mock=true",
        ]
    )
    args = build_megatron_args(cfg)

    assert "--mock-data" in args
    assert "--data-path" not in args
    assert "--data-cache-path" not in args
    assert "--split" not in args
    assert "--tokenizer-model" in args
