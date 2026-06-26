"""Unit tests for translating slm configs into Megatron CLI args."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from launchers.submit import _parse_overrides
from src.utils.megatron_args import build_megatron_args

# Minimal model dict accepted by ``_model_args`` for unit tests that exercise a
# single emission path without composing a full Hydra config.
_MIN_MODEL = {
    "num_layers": 2,
    "hidden_size": 64,
    "ffn_hidden_size": 128,
    "num_attention_heads": 4,
    "num_query_groups": 4,
    "head_dim": 16,
    "seq_length": 128,
    "normalization": "RMSNorm",
    "norm_epsilon": 1e-6,
    "positional_encoding": "rope",
    "rotary_base": 10000,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "init_method_std": 0.02,
    "tie_embeddings": True,
    "activation": "SwiGLU",
}


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


def test_recompute_args_emitted_only_when_set():
    from src.utils.megatron_args import _model_args

    def emit(extra):
        cfg = OmegaConf.create({"base": {"model": {**_MIN_MODEL, **extra}}})
        return _args_to_map(_model_args(cfg))

    # Off by default — no recompute flags.
    assert "--recompute-granularity" not in emit({})

    # 'full' emits granularity + method + num-layers.
    full = emit(
        {"recompute_granularity": "full", "recompute_method": "block", "recompute_num_layers": 2}
    )
    assert full["--recompute-granularity"] == "full"
    assert full["--recompute-method"] == "block"
    assert full["--recompute-num-layers"] == "2"

    # 'selective' emits granularity only (method/num-layers are full-only).
    sel = emit({"recompute_granularity": "selective"})
    assert sel["--recompute-granularity"] == "selective"
    assert "--recompute-method" not in sel


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


def test_poet_default_path_omits_distributed_optimizer_under_dist_cluster():
    # POET's optimizer builder (src/optim/poet.py get_megatron_poet_optimizer)
    # rejects the Megatron distributed optimizer UNCONDITIONALLY on every path
    # (the raise precedes the stock-Adam / POETAdam / Lie / Muon builders). So the
    # launcher must NOT request one even when the cluster sets
    # distributed_optimizer=true, else the flag is emitted but the optimizer build
    # hard-crashes ("POET optimizer does not support distributed optimizer").
    # Regression guard for that launcher/optimizer agreement.
    cfg = _parse_overrides(["experiment=optim/poet", "cluster=h100_de"])
    args = build_megatron_args(cfg)
    assert "--use-distributed-optimizer" not in args
    assert "--overlap-grad-reduce" not in args
    assert "--overlap-param-gather" not in args


def test_adamw_emits_distributed_optimizer_under_dist_cluster():
    # adamw is the stock Megatron path and DOES drive the sharded distributed
    # optimizer; the poet fix above must not have disabled it for adamw.
    cfg = _parse_overrides(["experiment=optim/adam", "cluster=h100_de"])
    args = build_megatron_args(cfg)
    assert "--use-distributed-optimizer" in args
    assert "--overlap-grad-reduce" in args
    assert "--overlap-param-gather" in args


def test_poet_custom_poetadam_path_omits_distributed_optimizer():
    # The custom POETAdam (ChainedOptimizer) path builds its own optimizer and
    # does not drive the sharded distributed optimizer — keep it off.
    cfg = _parse_overrides(
        ["experiment=optim/poet", "cluster=h100_de", "optim.poet.use_poet_adam=true"]
    )
    args = build_megatron_args(cfg)
    assert "--use-distributed-optimizer" not in args


def test_poet_muon_q_path_omits_distributed_optimizer():
    # Muon-on-Q explicitly raises on the distributed optimizer (dev-only), so its
    # argv must not request one (merge_period=0 is the no-reset regime Muon needs).
    cfg = _parse_overrides(
        [
            "experiment=optim/poet",
            "cluster=h100_de",
            "optim.poet.q_optimizer=muon",
            "optim.poet.merge_period=0",
        ]
    )
    args = build_megatron_args(cfg)
    assert "--use-distributed-optimizer" not in args


def test_poet_args_use_slm_optimizer_and_keep_megatron_optimizer_adam():
    cfg = _parse_overrides(["experiment=optim/poet"])
    args = _args_to_map(build_megatron_args(cfg))

    assert args["--optimizer"] == "adam"
    assert args["--slm-optimizer"] == "poet"
    assert args["--poet"] is True
    # POET takes EITHER --poet-block-size (bs) OR --poet-block-count (bc), never
    # both. The optim/poet experiment may set either, so assert the contract
    # (exactly one, positive value) rather than hardcoding one parameterization.
    has_bs, has_bc = "--poet-block-size" in args, "--poet-block-count" in args
    assert has_bs ^ has_bc, "poet must emit exactly one of block-size / block-count"
    assert int(args["--poet-block-size" if has_bs else "--poet-block-count"]) > 0
    assert args["--poet-merge-period"] == str(cfg.optim.poet.merge_period)


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


def _poet_cfg(poet_overrides):
    return OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 200,
                    "scale": 1.0,
                    **poet_overrides,
                },
            }
        }
    )


def test_poet_argv_emits_block_size_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-block-size" in args
    assert "--poet-block-count" not in args
    assert args[args.index("--poet-block-size") + 1] == "256"


def test_poet_argv_emits_block_count_when_set():
    from src.utils.megatron_args import _optimizer_args

    # block_count takes precedence; block_size must NOT be emitted alongside it.
    args = _optimizer_args(_poet_cfg({"block_size": 256, "block_count": 8}))
    assert "--poet-block-count" in args
    assert "--poet-block-size" not in args
    assert args[args.index("--poet-block-count") + 1] == "8"


def test_wandb_entity_omitted_when_unset():
    # Default entity is null -> do NOT emit --wandb-entity, so wandb falls back
    # to the account's personal namespace (avoids "entity not found").
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
    assert "--wandb-entity" not in args


def test_wandb_entity_passed_through_when_set():
    # When a team is configured, it must reach Megatron.
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
            "wandb.entity=some_team",
        ]
    )
    args = _args_to_map(build_megatron_args(cfg))
    assert args["--wandb-entity"] == "some_team"


def test_scheduler_defaults_to_cosine_block():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    assert cfg.scheduler.type == "cosine"
    assert float(cfg.scheduler.warmup_fraction) == 0.01


def test_scheduler_override_selects_wsd():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    assert cfg.scheduler.type == "wsd"
    assert float(cfg.scheduler.wsd_decay_fraction) == 0.2


def test_cosine_scheduler_emits_warmup_fraction_and_min_lr():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=cosine",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-decay-style"] == "cosine"
    assert m["--lr-warmup-fraction"] == "0.01"
    # champion optim.lr = 1.0e-3, min_lr_ratio = 0.1
    assert m["--min-lr"] == str(1.0e-3 * 0.1)
    assert "--lr-decay-step-ratio" not in m


def test_poet_argv_emits_lie_ortho_knobs():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "q_optimizer": "lie_ortho",
                "lie_ortho_c": 0.02,
                "lie_ortho_method": "spectral",
                "lie_ortho_ns_steps": 20,
                "lie_ortho_use_second_moment": True,
                "lie_ortho_angle_dim_exp": -0.5,
            }
        )
    )
    assert args[args.index("--poet-q-optimizer") + 1] == "lie_ortho"
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.02"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "spectral"
    assert args[args.index("--poet-lie-ortho-ns-steps") + 1] == "20"
    assert "--poet-lie-ortho-use-second-moment" in args
    assert args[args.index("--poet-lie-ortho-angle-dim-exp") + 1] == "-0.5"


def test_poet_argv_lie_ortho_defaults():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho-use-second-moment" not in args
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.01"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "muon"


def test_poet_argv_emits_lie_ortho_distributed():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "q_optimizer": "lie_ortho", "lie_ortho_distributed": True})
    )
    assert "--poet-lie-ortho-distributed" in args


def test_poet_argv_omits_lie_ortho_distributed_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho-distributed" not in args


def test_poet_argv_emits_lie_ortho_decorrelate():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "q_optimizer": "lie_ortho",
                "lie_ortho_decorrelate": True,
                "lie_ortho_decorrelate_mode": "symmetric",
                "lie_ortho_decorrelate_lambda": 0.5,
                "lie_ortho_decorrelate_renorm": True,
                "lie_ortho_decorrelate_cos_threshold": 0.3,
            }
        )
    )
    assert "--poet-lie-ortho-decorrelate" in args
    assert args[args.index("--poet-lie-ortho-decorrelate-mode") + 1] == "symmetric"
    assert args[args.index("--poet-lie-ortho-decorrelate-lambda") + 1] == "0.5"
    assert args[args.index("--poet-lie-ortho-decorrelate-cos-threshold") + 1] == "0.3"
    assert "--poet-lie-ortho-decorrelate-renorm" in args


def test_poet_argv_decorrelate_renorm_off_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "q_optimizer": "lie_ortho", "lie_ortho_decorrelate": True})
    )
    # renorm is a store_true flag — absent unless explicitly requested.
    assert "--poet-lie-ortho-decorrelate-renorm" not in args
    assert args[args.index("--poet-lie-ortho-decorrelate-lambda") + 1] == "1.0"


def test_poet_argv_omits_lie_ortho_decorrelate_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho-decorrelate" not in args


def test_poet_argv_emits_lie_ortho_update_rms_knobs():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "scale": 1.0,
                "q_optimizer": "lie_ortho_update_rms",
                "lie_alternating": True,
                "lie_ortho_update_rms": 0.25,
                "lie_ortho_max_angle": 0.02,
                "lie_ortho_rms_mode": "weight",
            }
        )
    )
    assert args[args.index("--poet-q-optimizer") + 1] == "lie_ortho_update_rms"
    assert args[args.index("--poet-lie-ortho-update-rms") + 1] == "0.25"
    assert args[args.index("--poet-lie-ortho-max-angle") + 1] == "0.02"
    assert args[args.index("--poet-lie-ortho-rms-mode") + 1] == "weight"
    assert "--poet-lie-alternating" in args


def test_poet_lie_orth_update_rms_yaml_emits_expected_knobs():
    cfg = _parse_overrides(["experiment=optim/poet_lie_orth_update_rms"])
    args = _args_to_map(build_megatron_args(cfg))
    assert args["--poet-q-optimizer"] == "lie_ortho_update_rms"
    assert args["--poet-scale"] == "1.0"
    assert args["--poet-lie-ortho-update-rms"] == "0.2"
    assert args["--poet-lie-ortho-max-angle"] == "0.024"
    assert args["--poet-lie-ortho-rms-mode"] == "weight"
    assert args["--poet-lie-alternating"] is True


def test_poet_lie_orth_update_rms_rejects_poet_scale():
    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match=r"set optim\.poet\.scale=1\.0"):
        _optimizer_args(
            _poet_cfg(
                {
                    "block_count": 1,
                    "merge_period": 1,
                    "scale": 0.5,
                    "q_optimizer": "lie_ortho_update_rms",
                    "lie_alternating": True,
                }
            )
        )


def test_wsd_scheduler_emits_wsd_flags():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-decay-style"] == "WSD"
    assert m["--lr-wsd-decay-style"] == "cosine"
    assert "--lr-wsd-decay-samples" in m


def test_ngpt_forces_zero_lr_warmup():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=cosine",
            "experiment=arch/ngpt",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--lr-warmup-samples"] == "0"
    assert "--lr-warmup-fraction" not in m


def test_decay_only_resume_emits_finetune_and_override():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd_decay_only",
            "experiment=champion",
            "training_regime=final_wsd_decay_only",
            "cluster=h800_cn",
            "training.decay_tokens=1200000000",
            "training.stable_checkpoint_dir=/tmp/stable_ckpt",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--finetune"] is True
    assert m["--override-opt-param-scheduler"] is True
    assert m["--load"] == "/tmp/stable_ckpt"
    assert m["--lr-decay-style"] == "WSD"
    # whole run is the anneal: warmup 0, wsd tail == total decay samples.
    # 300m scale uses seq_length=256, so samples = decay_tokens // 256.
    assert m["--lr-warmup-fraction"] == "0.0"
    assert m["--lr-wsd-decay-samples"] == str(1_200_000_000 // 256)


def _run_name(experiment: str) -> str:
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            f"experiment={experiment}",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    return _args_to_map(build_megatron_args(cfg))["--wandb-exp-name"]


def test_wandb_run_name_has_lr_and_no_seed():
    # champion optim.lr = 1.0e-3; display name is "adam"; no "-s<seed>" suffix.
    # Megatron's --wandb-exp-name now carries the shared "[<backend>] " prefix.
    assert _run_name("champion") == "[megatron] adam-llama3-300m-lr0.001"


def test_wandb_run_name_muon_shows_adam_lr_and_muon_lr():
    # muon_hybrid: lr = Adam-side (optim.adam.lr = 1.0e-3),
    # plus muon_lr = Muon-side (optim.muon.lr = 2.0e-3).
    assert _run_name("optim/muon_hybrid") == "[megatron] muon-llama3-300m-lr0.001-muon_lr0.002"


def test_wandb_run_name_poet_appends_block_param():
    # poet's wandb name carries the [megatron] prefix and appends a block
    # parameterization segment (bs<n> for block_size, bc<n> for block_count).
    # Assert the stable SHAPE, not the exact lr/block values (those track the
    # poet experiment config, which evolves independently of the naming logic).
    name = _run_name("optim/poet")
    assert name.startswith("[megatron] poet-llama3-300m-")
    assert any(seg[:2] in ("bs", "bc") and seg[2:].isdigit() for seg in name.split("-"))


def test_wandb_run_name_poet_block_count_overrides_block_size():
    # block_count logic lives in the shared wandb_base_name (un-prefixed canonical).
    from src.utils.wandb_naming import wandb_base_name

    cfg = OmegaConf.create(
        {
            "experiment": {"name": "poet"},
            "base": {"family": "llama3", "scale": "300m"},
            "optim": {"type": "poet", "lr": 3.0e-4, "poet": {"block_count": 8}},
        }
    )
    assert wandb_base_name(cfg) == "poet-llama3-300m-bc8-lr0.0003-scale1"


def test_unfuse_flags_default_on_for_all_train_script_experiments():
    # poet/adam/muon/ngpt all default base.model.unfuse_qkv/unfuse_fc1 = true.
    cases = {
        "poet": ["experiment=optim/poet"],
        "adam": ["experiment=optim/adam"],
        "muon": ["experiment=optim/muon_hybrid"],
        "ngpt": ["base/family=llama3", "base/scale=300m", "experiment=arch/ngpt"],
    }
    for label, overrides in cases.items():
        args = build_megatron_args(_parse_overrides(overrides))
        assert "--unfuse-qkv" in args, label
        assert "--unfuse-fc1" in args, label


def test_unfuse_can_be_disabled_per_run():
    # optim/poet now defaults head_aligned_attn=true, which requires unfused q/k/v;
    # disable it alongside unfuse so this isolates the unfuse-flag plumbing.
    cfg = _parse_overrides(
        [
            "experiment=optim/poet",
            "optim.poet.head_aligned_attn=false",
            "base.model.unfuse_qkv=false",
            "base.model.unfuse_fc1=false",
        ]
    )
    args = build_megatron_args(cfg)
    assert "--unfuse-qkv" not in args
    assert "--unfuse-fc1" not in args


def test_muon_kimi_argv_routes_through_adam_and_sets_muon_knobs():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "muon_kimi",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "muon_momentum": 0.95,
                "muon_use_nesterov": True,
                "muon_num_ns_steps": 5,
                "adam": {"betas": [0.9, 0.95], "eps": 1.0e-8},
            }
        }
    )
    args = _optimizer_args(cfg)
    amap = {args[i]: args[i + 1] for i in range(0, len(args) - 1)}
    assert amap["--optimizer"] == "adam"
    assert amap["--slm-optimizer"] == "muon_kimi"
    assert amap["--muon-momentum"] == "0.95"
    assert amap["--muon-num-ns-steps"] == "5"
    assert amap["--adam-beta1"] == "0.9"
    assert amap["--adam-beta2"] == "0.95"
    assert "--muon-use-nesterov" in args


def test_pion_argv_routes_through_adam_and_sets_pion_knobs():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "pion",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "pion_scaling": "rms",
                "pion_rms": 0.2,
                "pion_update_side": "alternate",
                "pion_momentum": "transported_ambient_ambient",
                "pion_degree": 2,
                "pion_beta1": 0.9,
                "pion_beta2": 0.95,
                "pion_use_second_momentum": False,
                "adam": {"betas": [0.9, 0.95], "eps": 1.0e-8},
            }
        }
    )
    args = _optimizer_args(cfg)
    amap = {args[i]: args[i + 1] for i in range(0, len(args) - 1)}
    assert amap["--optimizer"] == "adam"
    assert amap["--slm-optimizer"] == "pion"
    assert amap["--pion-scaling"] == "rms"
    assert amap["--pion-rms"] == "0.2"
    assert amap["--pion-update-side"] == "alternate"
    assert amap["--pion-momentum"] == "transported_ambient_ambient"
    assert amap["--pion-degree"] == "2"
    assert amap["--pion-beta1"] == "0.9"
    assert amap["--pion-beta2"] == "0.95"
    assert amap["--adam-beta1"] == "0.9"
    assert amap["--adam-beta2"] == "0.95"
    assert "--pion-use-second-momentum" not in args


def test_pion_argv_emits_second_momentum_flag_when_enabled():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "pion",
                "lr": 1e-3,
                "weight_decay": 0.1,
                "pion_use_second_momentum": True,
            }
        }
    )
    assert "--pion-use-second-momentum" in _optimizer_args(cfg)


def test_poet_argv_includes_parameterization_when_set():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                    "parameterization": "exp",
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert "--poet-parameterization" in args
    assert args[args.index("--poet-parameterization") + 1] == "exp"


def test_poet_argv_parameterization_defaults_to_cayley():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-parameterization") + 1] == "cayley"


def test_poet_argv_includes_q_optimizer_and_muon_knobs():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                    "q_optimizer": "muon",
                    "muon_theta": 0.2,
                    "muon_ns_steps": 5,
                    "muon_momentum": 0.95,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-q-optimizer") + 1] == "muon"
    # _optimizer_args stringifies all argv values (_sequence -> list[str]).
    assert args[args.index("--poet-muon-theta") + 1] == "0.2"


def test_poet_argv_q_optimizer_defaults_to_adam():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-q-optimizer") + 1] == "adam"


def test_poet_argv_emits_reinit_period_zero_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-reinit-period" in args
    assert args[args.index("--poet-reinit-period") + 1] == "0"


def test_poet_argv_emits_reinit_period_when_set():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256, "merge_period": 1, "reinit_period": 400}))
    assert args[args.index("--poet-merge-period") + 1] == "1"
    assert args[args.index("--poet-reinit-period") + 1] == "400"


def test_poet_argv_rejects_reinit_period_not_multiple_of_merge():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="multiple of"):
        _optimizer_args(_poet_cfg({"block_size": 256, "merge_period": 3, "reinit_period": 400}))


def test_poet0_experiment_yaml_sets_single_step_cadences():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet0.yaml")
    assert cfg.experiment.name == "poet0"
    assert cfg.optim.poet.merge_period == 1
    assert cfg.optim.poet.reinit_period == 400
    # poet0 keeps the stock optimizer (no Pion imports yet).
    assert cfg.optim.poet.use_poet_adam is False
    assert cfg.optim.poet.parameterization == "cayley"
    assert cfg.optim.poet.q_optimizer == "adam"


def test_poet_argv_emits_negative_reinit_period_without_validation_error():
    from src.utils.megatron_args import _optimizer_args

    # reinit_period < 0 (never reinit) must pass through and NOT trip the
    # "multiple of merge_period" validation (that guard is for positive periods).
    args = _optimizer_args(_poet_cfg({"block_size": 256, "merge_period": 1, "reinit_period": -1}))
    assert args[args.index("--poet-reinit-period") + 1] == "-1"


def test_poet_argv_emits_lie_args():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_size": 256,
                "q_optimizer": "lie_algebra",
                "lie_b1": 0.9,
                "lie_b2": 0.95,
                "lie_eps": 1e-8,
                "lie_v_mode": "elementwise",
            }
        )
    )
    assert args[args.index("--poet-q-optimizer") + 1] == "lie_algebra"
    assert args[args.index("--poet-lie-b1") + 1] == "0.9"
    assert args[args.index("--poet-lie-b2") + 1] == "0.95"
    assert args[args.index("--poet-lie-v-mode") + 1] == "elementwise"


def test_poet_argv_lie_args_default_when_unset():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert args[args.index("--poet-lie-v-mode") + 1] == "elementwise"
    assert args[args.index("--poet-q-optimizer") + 1] == "adam"


def test_poet_lie_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie.yaml")
    assert cfg.experiment.name == "poet_lie"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.merge_period == 1
    assert cfg.optim.poet.reinit_period == -1
    assert cfg.optim.poet.lie_v_mode == "elementwise"
    assert cfg.optim.poet.use_poet_adam is False


def test_poet_argv_emits_lie_alternating():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_size": 256,
                "q_optimizer": "lie_algebra",
                "lie_alternating": True,
                "lie_alternate_every": 2,
            }
        )
    )
    assert "--poet-lie-alternating" in args
    assert args[args.index("--poet-lie-alternate-every") + 1] == "2"


def test_poet_argv_omits_lie_alternating_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-alternating" not in args
    assert args[args.index("--poet-lie-alternate-every") + 1] == "1"


def test_poet_lie_alt_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_alt.yaml")
    assert cfg.experiment.name == "poet_lie_alt"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.lie_alternating is True
    assert cfg.optim.poet.lie_alternate_every == 1
    assert cfg.optim.poet.reinit_period == -1


def test_poet_argv_emits_lie_rms():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_size": 256,
                "q_optimizer": "lie_algebra",
                "lie_rms": True,
                "lie_rms_c": 0.3,
            }
        )
    )
    assert "--poet-lie-rms" in args
    assert args[args.index("--poet-lie-rms-c") + 1] == "0.3"


def test_poet_argv_omits_lie_rms_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-rms" not in args
    assert args[args.index("--poet-lie-rms-c") + 1] == "0.2"


def test_poet_lie_rms_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_rms.yaml")
    assert cfg.experiment.name == "poet_lie_rms"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.lie_rms is True
    assert cfg.optim.poet.lie_rms_c == 0.2
    assert cfg.optim.poet.lie_v_mode == "elementwise"


def test_poet_head_aligned_args_emitted_and_guard():
    import pytest
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    def make_cfg(head_aligned, head_resid_perm, unfuse_qkv):
        return OmegaConf.create(
            {
                "optim": {
                    "type": "poet",
                    "lr": 1e-3,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "poet": {
                        "block_count": 1,
                        "merge_period": 1,
                        "reinit_period": -1,
                        "scale": 0.5,
                        "init_type": "normalized",
                        "mup_alpha": 1.0,
                        "cache_mode": "none",
                        "parameterization": "cayley",
                        "q_optimizer": "lie_algebra",
                        "head_aligned_attn": head_aligned,
                        "head_resid_perm": head_resid_perm,
                    },
                },
                "base": {"model": {"unfuse_qkv": unfuse_qkv}},
            }
        )

    emitted = list(_optimizer_args(make_cfg(True, False, True)))
    assert "--poet-head-aligned-attn" in emitted
    assert "--poet-no-head-resid-perm" in emitted

    # Off by default -> neither flag emitted.
    off = list(_optimizer_args(make_cfg(False, True, True)))
    assert "--poet-head-aligned-attn" not in off
    assert "--poet-no-head-resid-perm" not in off

    # Guard: head-aligned without unfused qkv -> ValueError.
    with pytest.raises(ValueError, match="unfuse_qkv"):
        _optimizer_args(make_cfg(True, True, False))


def test_poet_lie_orth_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_orth.yaml")
    assert cfg.experiment.name == "poet_lie_orth"
    assert cfg.optim.poet.q_optimizer == "lie_ortho"
    assert cfg.optim.poet.lie_ortho_method == "muon"
    assert cfg.optim.poet.lie_ortho_c == 4
    assert cfg.optim.poet.lie_ortho_distributed is True


def test_poet_experiment_yamls_enable_lie_ortho_distributed():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    failures = []
    for path in sorted((root / "configs/experiments").rglob("*.yaml")):
        cfg = OmegaConf.load(path)
        if cfg.get("optim", {}).get("type") != "poet":
            continue
        if cfg.optim.poet.get("lie_ortho_distributed") is not True:
            failures.append(path.relative_to(root).as_posix())
    assert failures == []


def test_single_step_fast_requires_merge_period_one():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    # _poet_cfg defaults merge_period=200 -> single_step_fast must be rejected.
    with pytest.raises(ValueError, match="single_step_fast"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_fast": True}))


def test_single_step_fast_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "single_step_fast": True})
    )
    assert "--poet-single-step-fast" in args


def test_single_step_native_requires_merge_period_one():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_native"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_native": True}))


def test_single_step_native_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "single_step_native": True})
    )
    assert "--poet-single-step-native" in args


def test_single_step_x_requires_merge_period_one():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_x"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_x": True}))


def test_single_step_x_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_count": 1, "merge_period": 1, "single_step_x": True}))
    assert "--poet-single-step-x" in args


def test_single_step_x_alternating_emits_flag_when_valid():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "parameterization": "cayley",
                "q_optimizer": "lie_ortho",
                "single_step_x": True,
                "single_step_x_alternating": True,
                "head_aligned_attn": False,
                "train_output_rotation": True,
            }
        )
    )
    assert "--poet-single-step-x-alternating" in args


def test_single_step_x_alternating_requires_head_aligned_off():
    import pytest
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    # unfuse_qkv=true so the earlier head_aligned/unfuse guard passes and we reach
    # the single_step_x_alternating-specific head_aligned check.
    cfg = _poet_cfg(
        {
            "block_count": 1,
            "merge_period": 1,
            "parameterization": "cayley",
            "q_optimizer": "lie_ortho",
            "single_step_x": True,
            "single_step_x_alternating": True,
            "head_aligned_attn": True,
            "train_output_rotation": True,
        }
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"base": {"model": {"unfuse_qkv": True}}}))
    with pytest.raises(ValueError, match="head_aligned_attn"):
        _optimizer_args(cfg)


def test_single_step_x_with_lie_alternating_emits_both_flags():
    # Integrated path: single_step_x + lie_alternating is ALLOWED (only
    # single_step_x_alternating is mutually exclusive with lie_alternating).
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "parameterization": "cayley",
                "q_optimizer": "lie_ortho",
                "single_step_x": True,
                "lie_alternating": True,
                "single_step_x_alternating": False,
            }
        )
    )
    assert "--poet-single-step-x" in args
    assert "--poet-lie-alternating" in args
    assert "--poet-single-step-x-alternating" not in args


def test_poet_lie_orth_alt_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_orth_alt.yaml")
    assert cfg.experiment.name == "poet_lie_orth_alt"
    assert cfg.optim.poet.q_optimizer == "lie_ortho"
    assert cfg.optim.poet.single_step_x is True
    assert cfg.optim.poet.lie_alternating is True
    assert cfg.optim.poet.single_step_x_alternating is False
    assert cfg.optim.poet.head_aligned_attn is False
    assert cfg.optim.poet.reinit_period == -1


def test_head_resid_block_count_emits_flag():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = _poet_cfg(
        {
            "block_count": 1,
            "merge_period": 1,
            "parameterization": "cayley",
            "q_optimizer": "lie_ortho",
            "single_step_x": True,
            "head_aligned_attn": True,
            "head_resid_block_count": 4,
        }
    )
    # head_aligned_attn requires base.model.unfuse_qkv=true (an earlier guard).
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"base": {"model": {"unfuse_qkv": True}}}))
    args = _optimizer_args(cfg)
    assert "--poet-head-resid-block-count" in args
    assert args[args.index("--poet-head-resid-block-count") + 1] == "4"


def test_head_resid_block_count_requires_head_aligned():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="head_resid_block_count"):
        _optimizer_args(
            _poet_cfg(
                {
                    "block_count": 1,
                    "merge_period": 1,
                    "parameterization": "cayley",
                    "q_optimizer": "lie_ortho",
                    "single_step_x": True,
                    "head_aligned_attn": False,
                    "head_resid_block_count": 4,
                }
            )
        )


def test_rotary_percent_is_config_driven():
    from src.utils.megatron_args import _model_args

    cfg = OmegaConf.create({"base": {"model": _MIN_MODEL | {"rotary_percent": 0.25}}})
    args = _model_args(cfg)
    assert args[args.index("--rotary-percent") + 1] == "0.25"


def test_rotary_percent_defaults_to_one():
    from src.utils.megatron_args import _model_args

    cfg = OmegaConf.create({"base": {"model": _MIN_MODEL}})
    args = _model_args(cfg)
    assert args[args.index("--rotary-percent") + 1] == "1.0"


def test_sandwich_norm_flags_emitted_when_enabled():
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "use_sandwich_norm": True,
        "attn_post_norm_scale": 0.03,
        "ffn_post_norm_scale": 0.03,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--use-sandwich-norm" in args
    assert args[args.index("--attn-post-norm-scale") + 1] == "0.03"
    assert args[args.index("--ffn-post-norm-scale") + 1] == "0.03"


def test_sandwich_norm_flags_omitted_by_default():
    from src.utils.megatron_args import _model_args

    args = _model_args(OmegaConf.create({"base": {"model": _MIN_MODEL}}))
    assert "--use-sandwich-norm" not in args
    assert "--attn-post-norm-scale" not in args


def test_moe_router_fusion_and_layer_recompute_emitted():
    from src.utils.megatron_args import _model_args

    moe = {
        "enabled": True,
        "num_experts": 8,
        "layer_freq": "([1]*2)",
        "ffn_hidden_size": 128,
        "shared_expert_intermediate_size": 128,
        "router_load_balancing_type": "seq_aux_loss",
        "router_topk": 2,
        "token_dispatcher_type": "alltoall",
        "enable_deepep": False,
        "router_pre_softmax": False,
        "grouped_gemm": False,
        "aux_loss_coeff": 1e-4,
        "router_topk_scaling_factor": 2.5,
        "router_score_function": "sigmoid",
        "router_enable_expert_bias": True,
        "router_bias_update_rate": 1e-3,
        "router_dtype": "fp32",
        "permute_fusion": True,
        "router_fusion": True,
        "layer_recompute": True,
    }
    model = _MIN_MODEL | {"moe": moe}
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--moe-router-fusion" in args
    assert "--moe-layer-recompute" in args


def test_mtp_emitted_without_mla():
    # Review fix: MTP must emit for MQA (no MLA), where Huawei DeepSeek-3Bv2 uses it.
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "multi_latent_attention": False,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.3,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert args[args.index("--mtp-num-layers") + 1] == "1"
    assert args[args.index("--mtp-loss-scaling-factor") + 1] == "0.3"
    assert "--enable-experimental" in args


def test_mtp_still_emitted_with_mla():
    # Regression: the existing MLA path still emits MTP + experimental.
    from src.utils.megatron_args import _model_args

    model = _MIN_MODEL | {
        "multi_latent_attention": True,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_head_dim": 16,
        "qk_pos_emb_head_dim": 8,
        "v_head_dim": 16,
        "rotary_scaling_factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.1,
    }
    args = _model_args(OmegaConf.create({"base": {"model": model}}))
    assert "--mtp-num-layers" in args
    assert "--enable-experimental" in args


def test_total_tokens_suffix_string_accepted_without_resolve():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    cfg.training.total_tokens = "1B"  # as a CLI dotlist would leave it
    args = _args_to_map(build_megatron_args(cfg))
    # 60m uses seq_length 256: 1B tokens -> 3_906_250 samples
    assert args["--train-samples"] == str(1_000_000_000 // 256)
    assert args["--lr-decay-samples"] == str(1_000_000_000 // 256)


def test_fixed_total_tokens_pins_train_samples_across_scales_and_data():
    def args_for(scale: str, data: str) -> dict:
        cfg = _parse_overrides(
            [
                "base/family=llama3",
                f"base/scale={scale}",
                "experiment=champion",
                "training_regime=ablation_20x",
                "cluster=h800_cn",
                f"data={data}",
            ]
        )
        cfg.training.total_tokens = 1_000_000_000
        return _args_to_map(build_megatron_args(cfg))

    a = args_for("60m", "nemotron_cc_v2_llama31_8b")
    b = args_for("300m", "nemotron_cc_v2_llama31_8b")
    c = args_for("60m", "nemotron_cc_v2_scratch_qwen3")

    # (a) Same budget, near scales (both seq 256) -> identical sample count,
    # i.e. identical GPTDataset cache key -> no rebuild between the two.
    assert a["--train-samples"] == b["--train-samples"] == str(1_000_000_000 // 256)
    # (b) Different dataset/tokenizer -> rebuild (different path + cache dir),
    # but the token budget stays exactly as specified.
    assert c["--train-samples"] == a["--train-samples"]
    assert c["--data-path"] != a["--data-path"]
    assert c["--data-cache-path"] != a["--data-cache-path"]


def test_weight_norm_args_emits_flags_when_enabled():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _weight_norm_args

    training = OmegaConf.create(
        {
            "log_weight_norms": True,
            "log_weight_norms_interval": 50,
            "weight_norm_layers": "first,last",
        }
    )
    argv = _weight_norm_args(training)
    assert "--log-weight-norms" in argv
    assert argv[argv.index("--log-weight-norms-interval") + 1] == "50"
    assert argv[argv.index("--weight-norm-layers") + 1] == "first,last"


def test_weight_norm_args_omits_flags_by_default():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _weight_norm_args

    assert _weight_norm_args(OmegaConf.create({})) == []
    # bool false also emits nothing
    assert _weight_norm_args(OmegaConf.create({"log_weight_norms": False})) == []


def test_delta_w_args_emits_flags_when_enabled():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _delta_w_args

    training = OmegaConf.create(
        {
            "log_delta_w": True,
            "log_delta_w_interval": 50,
            "delta_w_layers": "first,last",
            "delta_w_max_targets": 3,
            "delta_w_spectral_max_dim": 64,
        }
    )
    argv = _delta_w_args(training)
    assert "--log-delta-w" in argv
    assert argv[argv.index("--log-delta-w-interval") + 1] == "50"
    assert argv[argv.index("--delta-w-layers") + 1] == "first,last"
    assert argv[argv.index("--delta-w-max-targets") + 1] == "3"
    assert argv[argv.index("--delta-w-spectral-max-dim") + 1] == "64"


def test_delta_w_args_omits_flags_by_default():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _delta_w_args

    assert _delta_w_args(OmegaConf.create({})) == []
    assert _delta_w_args(OmegaConf.create({"log_delta_w": False})) == []


def _poet_moe_cfg(grouped_gemm: bool):
    """Minimal cfg exercising the POET arg-builder with an MoE model block.

    The poet sub-keys mirror test_poet_argv_includes_cache_mode (the minimal set
    _optimizer_args needs to complete). base.model.moe drives the new guard.
    """
    return OmegaConf.create(
        {
            "base": {"model": {"moe": {"enabled": True, "grouped_gemm": grouped_gemm}}},
            "optim": {
                "type": "poet",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "block_size": 256,
                    "cache_mode": "none",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 1,
                    "scale": 1.0,
                },
            },
        }
    )


def test_poet_rejects_grouped_gemm_experts():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="grouped_gemm"):
        _optimizer_args(_poet_moe_cfg(grouped_gemm=True))


def test_poet_allows_sequential_experts():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_moe_cfg(grouped_gemm=False))
    assert "--poet" in args  # arg build completes, no raise


def test_poet_guard_inert_without_moe():
    # No base.model.moe block at all -> guard must not fire (dense POET path).
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
                    "cache_mode": "none",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 1,
                    "scale": 1.0,
                },
            }
        }
    )
    assert "--poet" in _optimizer_args(cfg)


def _one_sided_cfg(side):
    return _poet_cfg(
        {
            "block_count": 1,
            "merge_period": 1,
            "parameterization": "cayley",
            "q_optimizer": "lie_ortho",
            "single_step_fast": True,
            "single_step_x": True,
            "single_step_x_alternating": False,
            "lie_alternating": False,
            "train_output_rotation": True,
            "single_step_x_one_sided": side,
        }
    )


def test_one_sided_emits_flag_when_valid():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_one_sided_cfg("in"))
    assert "--poet-single-step-x-one-sided" in args
    assert args[args.index("--poet-single-step-x-one-sided") + 1] == "in"


def test_one_sided_omitted_when_unset():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {"block_count": 1, "merge_period": 1, "single_step_x": True, "q_optimizer": "lie_ortho"}
        )
    )
    assert "--poet-single-step-x-one-sided" not in args


def test_one_sided_requires_single_step_x():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _one_sided_cfg("in")
    cfg.optim.poet.single_step_x = False
    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(cfg)


def test_one_sided_mutually_exclusive_with_alternating():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _one_sided_cfg("out")
    cfg.optim.poet.single_step_x_alternating = True
    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(cfg)


def test_one_sided_rejects_bad_value():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(_one_sided_cfg("left"))


def test_in_only_yaml_emits_one_sided_in():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet_lie_orth_in_only",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--poet-single-step-x-one-sided"] == "in"
    assert m["--poet-single-step-x"] is True
    assert "--poet-single-step-x-alternating" not in m


def test_out_only_yaml_emits_one_sided_out():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet_lie_orth_out_only",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--poet-single-step-x-one-sided"] == "out"
