"""Composition + arg-emission tests for the deepseek_v3_mqa family / deepseek_3bv2 scale."""

from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args


def _cfg():
    return _parse_overrides(
        ["base/family=deepseek_v3_mqa", "base/scale=deepseek_3bv2", "experiment=optim/adam"]
    )


def test_scale_resolves_mqa_and_sandwich():
    m = _cfg().base.model
    assert m.num_layers == 12
    assert m.hidden_size == 1280
    assert m.ffn_hidden_size == 7168
    assert m.num_attention_heads == 16
    assert m.head_dim == 384
    assert m.num_query_groups == 1
    assert m.multi_latent_attention is False
    assert m.use_sandwich_norm is True
    assert m.rotary_percent == 0.25
    assert m.moe.ffn_hidden_size == 896
    assert m.moe.router_topk == 6


def test_megatron_args_emit_mqa_sandwich_moe():
    args = build_megatron_args(_cfg())
    assert "--group-query-attention" in args
    assert args[args.index("--num-query-groups") + 1] == "1"
    assert args[args.index("--kv-channels") + 1] == "384"
    assert args[args.index("--rotary-percent") + 1] == "0.25"
    assert "--use-sandwich-norm" in args
    assert args[args.index("--attn-post-norm-scale") + 1] == "0.03"
    assert args[args.index("--moe-router-topk") + 1] == "6"
    assert args[args.index("--moe-ffn-hidden-size") + 1] == "896"
    assert "--multi-latent-attention" not in args


def test_embedding_init_std_emitted_to_match_huawei():
    # Huawei DeepSeek-3Bv2 sets --embedding-init-method-std 0.006 explicitly.
    args = build_megatron_args(_cfg())
    assert args[args.index("--embedding-init-method-std") + 1] == "0.006"


def test_sandwich_patch_listed_in_experiments():
    # poet is now included: sandwich_norm_apply owns only gpt_builders.gpt_builder
    # and stamps the config via a temporary wrapper inside the builder, so it no
    # longer collides with poet_unfuse_te_impl on core_transformer_config_from_args.
    for exp in ("optim/adam", "optim/muon_hybrid", "optim/poet"):
        cfg = _parse_overrides([f"experiment={exp}"])
        patches = list(cfg.experiment.patches)
        assert "sandwich_norm_apply" in patches, exp
