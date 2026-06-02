"""Translate resolved slm-research configs into Megatron GPT CLI args."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.scheduler import scheduler_args
from src.utils.wandb_naming import wandb_run_name


def _truthy(value: Any) -> bool:
    return bool(value)


def _add(args: list[str], flag: str, value: Any | None = None) -> None:
    args.append(flag)
    if value is not None:
        args.append(str(value))


def _maybe_bool(args: list[str], flag: str, value: Any) -> None:
    if _truthy(value):
        _add(args, flag)


def _sequence(values: Iterable[Any]) -> list[str]:
    return [str(v) for v in values]


def _model_args(cfg: DictConfig) -> list[str]:
    base = cfg.base
    model = base.model
    args: list[str] = []

    _add(args, "--use-mcore-models")
    if model.get("transformer_impl", None) is not None:
        _add(args, "--transformer-impl", model.transformer_impl)
        if str(model.transformer_impl) == "local":
            # Local impl uses torch.nn.LayerNorm via WrappedTorchNorm, which
            # asserts `not config.persist_layer_norm`. The TE-backed fused
            # persistent layernorm kernel isn't reachable from this path.
            _add(args, "--no-persist-layer-norm")
            # Megatron's local DotProductAttention dispatches into
            # FusedScaleMaskSoftmax, whose CUDA kernel on this cluster ships
            # PTX newer than the installed driver supports (raises
            # cudaErrorUnsupportedPtxVersion at the first forward step). Fall
            # back to the plain torch softmax path.
            _add(args, "--no-masked-softmax-fusion")
    _add(args, "--num-layers", model.num_layers)
    _add(args, "--hidden-size", model.hidden_size)
    _add(args, "--ffn-hidden-size", model.ffn_hidden_size)
    _add(args, "--num-attention-heads", model.num_attention_heads)
    if int(model.num_query_groups) != int(model.num_attention_heads):
        _add(args, "--group-query-attention")
        _add(args, "--num-query-groups", model.num_query_groups)
    _add(args, "--kv-channels", model.head_dim)
    _add(args, "--seq-length", model.seq_length)
    _add(args, "--max-position-embeddings", model.get("max_position_embeddings", model.seq_length))
    _add(args, "--position-embedding-type", model.positional_encoding)
    _add(args, "--rotary-base", model.rotary_base)
    _add(args, "--rotary-percent", 1.0)
    _add(args, "--attention-dropout", model.attention_dropout)
    _add(args, "--hidden-dropout", model.hidden_dropout)
    _add(args, "--normalization", model.normalization)
    _add(args, "--norm-epsilon", model.norm_epsilon)
    _add(args, "--init-method-std", model.init_method_std)
    # Default `flash` (not `fused`): TE's `fused` backend dispatches to cuDNN's
    # fused attention, which on this cluster's cu13 stack silently falls back
    # to an O(seq²) path for head_dim=64 + GQA + seq=4096 — it OOMs a 4xB200
    # (178 GiB/GPU) at first forward step even at mbs=32. Flash-attn 2.8.3 is
    # installed and is O(seq) memory, so we force it as the default. Override
    # per-experiment with base.model.attention_backend=auto|fused|local.
    _add(args, "--attention-backend", model.get("attention_backend", "flash"))
    _add(args, "--swiglu")
    _add(args, "--disable-bias-linear")
    # Architectural unfusing of fused linears (optimizer-agnostic). Applied by
    # the ``model_unfuse_linears`` patch when listed in experiment.patches.
    if bool(model.get("unfuse_qkv", False)):
        args.append("--unfuse-qkv")
    if bool(model.get("unfuse_fc1", False)):
        args.append("--unfuse-fc1")
    if not bool(model.tie_embeddings):
        _add(args, "--untie-embeddings-and-output-weights")

    if bool(model.get("qk_norm", False)):
        _add(args, "--qk-layernorm")

    if bool(model.get("multi_latent_attention", False)):
        _add(args, "--multi-latent-attention")
        for key, flag in (
            ("q_lora_rank", "--q-lora-rank"),
            ("kv_lora_rank", "--kv-lora-rank"),
            ("qk_head_dim", "--qk-head-dim"),
            ("qk_pos_emb_head_dim", "--qk-pos-emb-head-dim"),
            ("v_head_dim", "--v-head-dim"),
            ("rotary_scaling_factor", "--rotary-scaling-factor"),
            ("mscale", "--mscale"),
            ("mscale_all_dim", "--mscale-all-dim"),
            ("mtp_num_layers", "--mtp-num-layers"),
            ("mtp_loss_scaling_factor", "--mtp-loss-scaling-factor"),
        ):
            _add(args, flag, model[key])
        _add(args, "--enable-experimental")

    moe = model.get("moe", {})
    if bool(moe.get("enabled", False)):
        _add(args, "--num-experts", moe.num_experts)
        _add(args, "--moe-layer-freq", moe.layer_freq)
        _add(args, "--moe-ffn-hidden-size", moe.ffn_hidden_size)
        _add(args, "--moe-shared-expert-intermediate-size", moe.shared_expert_intermediate_size)
        _add(args, "--moe-router-load-balancing-type", moe.router_load_balancing_type)
        _add(args, "--moe-router-topk", moe.router_topk)
        _add(args, "--moe-token-dispatcher-type", moe.token_dispatcher_type)
        _maybe_bool(args, "--moe-enable-deepep", moe.enable_deepep)
        _maybe_bool(args, "--moe-router-pre-softmax", moe.router_pre_softmax)
        _maybe_bool(args, "--moe-grouped-gemm", moe.grouped_gemm)
        _add(args, "--moe-aux-loss-coeff", moe.aux_loss_coeff)
        # DeepSeek-V3 n-group routing is optional; scales that disable it (e.g.
        # the DeepSeek-3B recipe) set these to null and we omit the flags.
        if moe.get("router_group_topk", None) is not None:
            _add(args, "--moe-router-group-topk", moe.router_group_topk)
        if moe.get("router_num_groups", None) is not None:
            _add(args, "--moe-router-num-groups", moe.router_num_groups)
        _add(args, "--moe-router-topk-scaling-factor", moe.router_topk_scaling_factor)
        _add(args, "--moe-router-score-function", moe.router_score_function)
        _maybe_bool(args, "--moe-router-enable-expert-bias", moe.router_enable_expert_bias)
        _add(args, "--moe-router-bias-update-rate", moe.router_bias_update_rate)
        _add(args, "--moe-router-dtype", moe.router_dtype)
        _maybe_bool(args, "--moe-permute-fusion", moe.permute_fusion)

    return args


def _training_args(cfg: DictConfig) -> list[str]:
    training = cfg.training
    optim = cfg.optim
    model = cfg.base.model
    seq_length = int(model.seq_length)
    global_batch_size = int(training.global_batch_size)
    micro_batch_raw = training.get("micro_batch_size", None)
    if micro_batch_raw is None:
        micro_batch_size = min(64, global_batch_size)
    else:
        micro_batch_size = int(micro_batch_raw)

    if bool(training.get("resume_from_stable_stage", False)):
        decay_tokens = training.get("decay_tokens", None)
        if decay_tokens is None:
            raise ValueError(
                "resume_from_stable_stage requires training.decay_tokens "
                "(e.g. training.decay_tokens=1_200_000_000)"
            )
        total_tokens = int(decay_tokens)
    else:
        total_tokens = int(training.get("total_tokens", 0)) or (
            int(training.get("tokens_per_param", 20)) * int(cfg.base.non_embedding_params)
        )

    peak_lr = optim.get("lr", optim.get("adam", {}).get("lr", 1.0e-3))
    force_no_warmup = bool(cfg.optim.get("ngpt", {}).get("no_warmup", False))

    args: list[str] = []
    _add(args, "--micro-batch-size", micro_batch_size)
    _add(args, "--global-batch-size", global_batch_size)
    _add(args, "--train-samples", total_tokens // seq_length)
    _add(args, "--lr-decay-samples", total_tokens // seq_length)
    _add(args, "--lr", peak_lr)
    args.extend(
        scheduler_args(
            cfg.scheduler,
            peak_lr=float(peak_lr),
            total_tokens=total_tokens,
            seq_length=seq_length,
            force_no_warmup=force_no_warmup,
        )
    )
    _add(args, "--clip-grad", training.get("clip_grad", 1.0))
    _add(args, "--weight-decay", optim.get("weight_decay", 0.1))
    _add(args, "--bf16")
    _add(args, "--cross-entropy-loss-fusion")
    _add(args, "--calculate-per-token-loss")
    return args


def _optimizer_args(cfg: DictConfig) -> list[str]:
    optim = cfg.optim
    kind = str(optim.type)

    if kind == "adamw":
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
                "--slm-optimizer",
                "adamw",
            ]
        )

    if kind == "muon_hybrid":
        muon = optim.muon
        return _sequence(
            [
                "--optimizer",
                "muon",
                "--slm-optimizer",
                "muon",
                "--muon-momentum",
                optim.get("muon_momentum", 0.95),
                "--muon-num-ns-steps",
                muon.ns_steps,
                "--muon-scale-mode",
                optim.get("muon_scale_mode", "spectral"),
                "--muon-tp-mode",
                optim.get("muon_tp_mode", "blockwise"),
            ]
        )

    if kind == "muon_kimi":
        adam = optim.get("adam", {})
        betas = adam.get("betas", [0.9, 0.95])
        argv = [
            "--optimizer",
            "adam",
            "--slm-optimizer",
            "muon_kimi",
            "--muon-momentum",
            optim.get("muon_momentum", 0.95),
            "--muon-num-ns-steps",
            optim.get("muon_num_ns_steps", 5),
            "--adam-beta1",
            betas[0],
            "--adam-beta2",
            betas[1],
            "--adam-eps",
            adam.get("eps", 1.0e-8),
        ]
        if bool(optim.get("muon_use_nesterov", True)):
            argv.append("--muon-use-nesterov")
        return _sequence(argv)

    if kind == "poet":
        poet = optim.poet
        # block_count (decoupled block sizes) takes precedence over block_size.
        block_count = poet.get("block_count", None)
        if block_count is not None:
            block_args = ["--poet-block-count", block_count]
        else:
            block_args = ["--poet-block-size", poet.block_size]
        poet_args = [
            "--optimizer",
            "adam",
            "--slm-optimizer",
            "poet",
            "--poet",
            *block_args,
            "--poet-init-type",
            poet.init_type,
            "--poet-mup-alpha",
            poet.mup_alpha,
            "--poet-merge-period",
            poet.merge_period,
            "--poet-scale",
            poet.scale,
            "--poet-cache-mode",
            poet.get("cache_mode", "none"),
            "--poet-parameterization",
            poet.get("parameterization", "cayley"),
            "--poet-q-optimizer",
            poet.get("q_optimizer", "adam"),
            "--poet-muon-theta",
            poet.get("muon_theta", 0.1),
            "--poet-muon-ns-steps",
            poet.get("muon_ns_steps", 5),
            "--poet-muon-momentum",
            poet.get("muon_momentum", 0.95),
            "--adam-beta1",
            optim.betas[0],
            "--adam-beta2",
            optim.betas[1],
            "--adam-eps",
            optim.eps,
        ]
        # store_true flag: emit only when the custom POETAdam path is requested
        # (default = stock Megatron-Adam path).
        if poet.get("use_poet_adam", False):
            poet_args.append("--poet-use-poet-adam")
        # store_true: freeze the output-side rotation (train the input rotation only).
        if not poet.get("train_output_rotation", True):
            poet_args.append("--poet-freeze-output-rotation")
        return _sequence(poet_args)

    if kind == "ngpt_adamw":
        ng = optim.get("ngpt", {})
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--slm-optimizer",
                "ngpt_adamw",
                "--ngpt",
                "--ngpt-alpha-init",
                float(ng.get("alpha_init", 0.05)),
                "--ngpt-sqk-init",
                float(ng.get("sqk_init", 1.0)),
                "--ngpt-suv-init",
                float(ng.get("suv_init", 1.0)),
                "--ngpt-sz-init",
                float(ng.get("sz_init", 1.0)),
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
            ]
            + (["--ngpt-no-warmup"] if bool(ng.get("no_warmup", True)) else [])
        )

    raise ValueError(f"Unsupported optimizer type {kind!r}")


def _parallel_args(cfg: DictConfig) -> list[str]:
    parallelism = cfg.parallelism
    args: list[str] = []
    _add(args, "--tensor-model-parallel-size", parallelism.get("tp", 1))
    _add(args, "--pipeline-model-parallel-size", parallelism.get("pp", 1))
    if bool(parallelism.get("sequence_parallel", True)):
        _add(args, "--sequence-parallel")
    if cfg.optim.type == "adamw" and bool(parallelism.get("distributed_optimizer", False)):
        _add(args, "--use-distributed-optimizer")
        _add(args, "--overlap-grad-reduce")
        _add(args, "--overlap-param-gather")
    return args


def _data_args(cfg: DictConfig) -> list[str]:
    data = cfg.data
    args: list[str] = []
    _add(args, "--data-path", data.path)
    _add(args, "--tokenizer-type", data.tokenizer_type)
    _add(args, "--tokenizer-model", data.tokenizer_model)
    _add(args, "--vocab-size", data.vocab_size)
    _add(args, "--data-cache-path", f"runs/_data_cache/{data.name}")
    _add(args, "--split", data.split)
    if bool(data.no_mmap_bin_files):
        _add(args, "--no-mmap-bin-files")
    if bool(data.no_create_attention_mask_in_dataloader):
        _add(args, "--no-create-attention-mask-in-dataloader")
    _add(args, "--num-workers", data.num_workers)
    return args


def _logging_args(cfg: DictConfig) -> list[str]:
    derived = cfg.get("_derived", {})
    archive = derived.get("run_dir", "runs/pending") if hasattr(derived, "get") else "runs/pending"
    training = cfg.training
    resume = bool(training.get("resume_from_stable_stage", False))
    load_dir = (
        str(training.get("stable_checkpoint_dir"))
        if resume and training.get("stable_checkpoint_dir", None) is not None
        else f"{archive}/checkpoints"
    )
    save_enabled = bool(training.get("save_enabled", True))
    args = _sequence(
        [
            "--log-interval",
            cfg.training.get("log_interval", 10),
            "--eval-iters",
            cfg.training.get("eval_iters", 32),
            "--eval-interval",
            cfg.training.get("eval_interval", 500),
            "--log-throughput",
            "--tensorboard-dir",
            f"{archive}/tensorboard",
            "--ckpt-format",
            cfg.training.get("ckpt_format", "torch_dist"),
            "--distributed-timeout-minutes",
            60,
            "--load",
            load_dir,
            "--wandb-project",
            cfg.wandb.project,
            "--wandb-exp-name",
            wandb_run_name(cfg),
        ]
    )
    # Checkpoint saving. With training.save_enabled=false we omit --save
    # entirely, so Megatron writes NO checkpoints — including the implicit
    # end-of-run save (every save_checkpoint() call is guarded by
    # `if args.save`). When enabled, --save-interval defaults to an
    # effectively-never value so only an explicit training.save_interval=N
    # checkpoints periodically.
    if save_enabled:
        _add(args, "--save-interval", cfg.training.get("save_interval", 1_000_000_000))
        _add(args, "--save", f"{archive}/checkpoints")
    # Only force an entity when one is configured. An empty/null entity lets
    # wandb fall back to the logged-in account's default (personal) namespace,
    # avoiding "entity ... not found" when the configured team is inaccessible.
    entity = cfg.wandb.get("entity", None)
    if entity:
        _add(args, "--wandb-entity", str(entity))
    if resume:
        _add(args, "--finetune")
        _add(args, "--override-opt-param-scheduler")
    return args


def build_megatron_args(cfg: DictConfig) -> list[str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    args: list[str] = []
    args.extend(_model_args(cfg))
    args.extend(_training_args(cfg))
    args.extend(_optimizer_args(cfg))
    args.extend(_parallel_args(cfg))
    args.extend(_data_args(cfg))
    args.extend(_logging_args(cfg))
    return args
