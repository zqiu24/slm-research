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
    assert _run_name("champion") == "adam-llama3-300m-lr0.001"


def test_wandb_run_name_muon_shows_adam_lr_and_muon_lr():
    # muon_hybrid: lr = Adam-side (optim.adam.lr = 1.0e-3),
    # plus muon_lr = Muon-side (optim.muon.lr = 2.0e-3).
    assert _run_name("optim/muon_hybrid") == "muon-llama3-300m-lr0.001-muon_lr0.002"


def test_wandb_run_name_poet_appends_block_param():
    # poet optim.lr = 3.0e-4; default uses block_size=256 (no block_count).
    name = _run_name("optim/poet")
    assert name.startswith("poet-llama3-300m-lr0.0003")
    assert name.endswith("-bs256")


def test_wandb_run_name_poet_block_count_overrides_block_size():
    from src.utils.megatron_args import _wandb_run_name

    cfg = OmegaConf.create(
        {
            "experiment": {"name": "poet"},
            "base": {"family": "llama3", "scale": "300m"},
            "optim": {"type": "poet", "lr": 3.0e-4, "poet": {"block_count": 8}},
        }
    )
    assert _wandb_run_name(cfg) == "poet-llama3-300m-lr0.0003-bc8"


def test_poet_experiment_emits_unfuse_flags_by_default():
    # The poet experiment sets base.model.unfuse_qkv/unfuse_fc1 = true.
    cfg = _parse_overrides(["experiment=optim/poet"])
    args = build_megatron_args(cfg)
    assert "--unfuse-qkv" in args
    assert "--unfuse-fc1" in args


def test_adam_experiment_omits_unfuse_flags():
    cfg = _parse_overrides(["experiment=optim/adam"])
    args = build_megatron_args(cfg)
    assert "--unfuse-qkv" not in args
    assert "--unfuse-fc1" not in args


def test_unfuse_flags_emitted_from_base_model_for_any_experiment():
    # Architectural: turning it on for a non-POET experiment also emits the flags.
    cfg = _parse_overrides(
        ["experiment=optim/adam", "base.model.unfuse_qkv=true", "base.model.unfuse_fc1=true"]
    )
    args = build_megatron_args(cfg)
    assert "--unfuse-qkv" in args
    assert "--unfuse-fc1" in args
