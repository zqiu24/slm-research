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
    # POET takes EITHER --poet-block-size (bs) OR --poet-block-count (bc), never
    # both. The optim/poet experiment may set either, so assert the contract
    # (exactly one, positive value) rather than hardcoding one parameterization.
    has_bs, has_bc = "--poet-block-size" in args, "--poet-block-count" in args
    assert has_bs ^ has_bc, "poet must emit exactly one of block-size / block-count"
    assert int(args["--poet-block-size" if has_bs else "--poet-block-count"]) > 0
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
    # Megatron's --wandb-exp-name now carries the shared "[<backend>] " prefix.
    assert _run_name("champion") == "[megatron] adam-llama3-300m-lr0.001"


def test_wandb_run_name_muon_shows_adam_lr_and_muon_lr():
    # muon_hybrid: lr = Adam-side (optim.adam.lr = 1.0e-3),
    # plus muon_lr = Muon-side (optim.muon.lr = 2.0e-3).
    assert _run_name("optim/muon_hybrid") == "[megatron] muon-llama3-300m-lr0.001-muon_lr0.002"


def test_wandb_run_name_poet_appends_block_param():
    # poet's wandb name carries the [megatron] prefix and appends a block
    # parameterization segment (-bs<n> for block_size, -bc<n> for block_count).
    # Assert the stable SHAPE, not the exact lr/block values (those track the
    # poet experiment config, which evolves independently of the naming logic).
    name = _run_name("optim/poet")
    assert name.startswith("[megatron] poet-llama3-300m-")
    block_seg = name.rsplit("-", 1)[-1]
    assert block_seg[:2] in ("bs", "bc") and block_seg[2:].isdigit()


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
    assert wandb_base_name(cfg) == "poet-llama3-300m-lr0.0003-bc8"


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
    cfg = _parse_overrides(
        ["experiment=optim/poet", "base.model.unfuse_qkv=false", "base.model.unfuse_fc1=false"]
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
