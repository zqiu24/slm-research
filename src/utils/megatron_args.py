"""Translate resolved slm-research configs into Megatron GPT CLI args."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.ladder_math import parse_token_count
from src.utils.scheduler import scheduler_args
from src.utils.wandb_naming import wandb_run_name

logger = logging.getLogger(__name__)


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
    transformer_impl = model.get("transformer_impl", None)
    # POET requires the local (unfused) impl: the GPT layer spec is selected
    # from args.transformer_impl (NOT config.transformer_impl), and POET cannot
    # replace fused TE LayerNormColumnParallelLinear. Force it here so a POET run
    # need not pass base.model.transformer_impl=local by hand (the runtime
    # poet_unfuse_te_impl patch only flips config, which the spec ignores).
    optim_cfg = cfg.get("optim", None)
    optim_type = str(optim_cfg.get("type", "")) if optim_cfg is not None else ""
    if optim_type == "poet" and str(transformer_impl) != "local":
        if transformer_impl is not None:
            logger.warning(
                "POET requires --transformer-impl local; overriding %r -> 'local'",
                str(transformer_impl),
            )
        transformer_impl = "local"
    if transformer_impl is not None:
        _add(args, "--transformer-impl", transformer_impl)
        if str(transformer_impl) == "local":
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
    if str(model.positional_encoding) == "rope":
        _add(args, "--rotary-base", model.rotary_base)
        _add(args, "--rotary-percent", model.get("rotary_percent", 1.0))
    _add(args, "--attention-dropout", model.attention_dropout)
    _add(args, "--hidden-dropout", model.hidden_dropout)
    _add(args, "--normalization", model.normalization)
    _add(args, "--norm-epsilon", model.norm_epsilon)
    # Activation recompute (off unless set). 'full' wraps each transformer layer
    # in tensor_parallel.checkpoint, re-running it in backward to free the
    # per-layer activations (memory <-> compute trade). 'method'/'num-layers'
    # only apply to 'full'.
    rg = model.get("recompute_granularity", None)
    if rg is not None:
        _add(args, "--recompute-granularity", rg)
        if str(rg) == "full":
            _add(args, "--recompute-method", model.get("recompute_method", "uniform"))
            _add(args, "--recompute-num-layers", model.get("recompute_num_layers", 1))
    # Zero-centered (1+w) RMSNorm (Gemma). --apply-layernorm-1p maps to the
    # config field layernorm_zero_centered_gamma (transformer_config.py:170
    # argparse_meta; pin core_v0.17.0).
    if bool(model.get("layernorm_zero_centered", False)):
        _add(args, "--apply-layernorm-1p")
    _add(args, "--init-method-std", model.init_method_std)
    # Optional separate embedding init std (Huawei DeepSeek-3Bv2 sets it explicitly).
    # Megatron defaults embedding init to --init-method-std when unset, so emit
    # only when the config provides it — keeps every other family unchanged.
    if model.get("embedding_init_method_std", None) is not None:
        _add(args, "--embedding-init-method-std", model.embedding_init_method_std)
    # Default `flash` (not `fused`): TE's `fused` backend dispatches to cuDNN's
    # fused attention, which on this cluster's cu13 stack silently falls back
    # to an O(seq²) path for head_dim=64 + GQA + seq=4096 — it OOMs a 4xB200
    # (178 GiB/GPU) at first forward step even at mbs=32. Flash-attn 2.8.3 is
    # installed and is O(seq) memory, so we force it as the default. Override
    # per-experiment with base.model.attention_backend=auto|fused|local.
    _add(args, "--attention-backend", model.get("attention_backend", "flash"))
    activation = str(model.get("activation", "SwiGLU"))
    if activation == "SwiGLU":
        _add(args, "--swiglu")
    elif activation == "squared_relu":
        _add(args, "--squared-relu")
    elif activation == "GeGLU":
        # Gemma-style gated GELU. --quick-geglu sets gated_linear_unit + the
        # sigmoid-approx quick_gelu (arguments.py:1676, pin core_v0.17.0); the
        # tanh-approx gelu_pytorch_tanh has no native flag (documented approx).
        _add(args, "--quick-geglu")
    else:
        raise ValueError(f"Unsupported model.activation {activation!r}")
    _add(args, "--disable-bias-linear")
    # Sandwich-norm (post-attn / post-MLP norm before the residual add). The
    # architecture is applied by the ``sandwich_norm_apply`` patch; here we only
    # emit the flags it reads off the parsed args.
    if bool(model.get("use_sandwich_norm", False)):
        args.append("--use-sandwich-norm")
        _add(args, "--attn-post-norm-scale", model.get("attn_post_norm_scale", 1.0))
        _add(args, "--ffn-post-norm-scale", model.get("ffn_post_norm_scale", 1.0))
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

    # Local/global sliding-window interleave (Gemma 3-style). Native Megatron
    # CLI args (arguments.py:2020/2023, pin core_v0.17.0): --window-size parses
    # "W,0" -> (W,0) via tuple_type; --window-attn-skip-freq takes int N (=
    # (N-1):1 SWA:full, so 6 -> 5 sliding : 1 global) or a list-expr string.
    sliding = model.get("sliding_window", {}) or {}
    if bool(sliding.get("enabled", False)):
        _add(args, "--window-size", f"{int(sliding.get('window', 1024))},0")
        _add(args, "--window-attn-skip-freq", sliding.get("skip_freq", 6))

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
        ):
            val = model.get(key)
            if val is not None:
                _add(args, flag, val)

    # MTP is independent of MLA (Huawei DeepSeek-3Bv2 uses MTP with MQA).
    if model.get("mtp_num_layers", None) is not None:
        _add(args, "--mtp-num-layers", model.mtp_num_layers)
        _add(args, "--mtp-loss-scaling-factor", model.get("mtp_loss_scaling_factor", 0.1))
    if (
        bool(model.get("multi_latent_attention", False))
        or model.get("mtp_num_layers", None) is not None
    ):
        _add(args, "--enable-experimental")

    # GatedDeltaNet linear attention (Qwen3-Next-style hybrids). All flags are
    # native Megatron CLI args auto-generated from TransformerConfig fields
    # (arguments.py:1655, pin core_v0.17.0); routed in gpt_builders.py via
    # args.experimental_attention_variant.
    gdn = model.get("gdn", {}) or {}
    if bool(gdn.get("enabled", False)):
        _add(args, "--experimental-attention-variant", "gated_delta_net")
        _add(args, "--linear-attention-freq", model.linear_attention_freq)
        _add(args, "--linear-num-key-heads", gdn.num_key_heads)
        _add(args, "--linear-key-head-dim", gdn.key_head_dim)
        _add(args, "--linear-num-value-heads", gdn.num_value_heads)
        _add(args, "--linear-value-head-dim", gdn.value_head_dim)
        _add(args, "--linear-conv-kernel-dim", gdn.get("conv_kernel_dim", 4))
        if "--enable-experimental" not in args:
            _add(args, "--enable-experimental")

    # Hybrid Mamba2 layer stacks (Nemotron-H-style). Megatron derives
    # num_layers and is_hybrid_model from the pattern; we validate coherence
    # here so a bad config fails at dry-run, not at rank startup. The mamba
    # path has no MTP support (pretrain_gpt.py docstring, pin core_v0.17.0).
    pattern = model.get("hybrid_layer_pattern", None)
    if pattern is not None:
        pattern = str(pattern)
        if len(pattern) != int(model.num_layers):
            raise ValueError(
                f"hybrid_layer_pattern length {len(pattern)} != num_layers {int(model.num_layers)}"
            )
        if model.get("mtp_num_layers", None):
            raise ValueError("MTP is not supported on the mamba/hybrid path")
        _add(args, "--hybrid-layer-pattern", pattern)
        # mamba_builder raises if args.spec is None ("You must provide a valid
        # Mamba layer spec via --spec", mamba_builders.py, pin core_v0.17.0).
        # --spec is nargs='*' (module path + attr); point it at the standard
        # mamba stack spec so the hybrid path builds without per-run flags.
        args.extend(["--spec", "megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec"])
        mamba = model.get("mamba", {}) or {}
        _add(args, "--mamba-state-dim", mamba.get("state_dim", 128))
        _add(args, "--mamba-head-dim", mamba.get("head_dim", 64))
        _add(args, "--mamba-num-groups", mamba.get("num_groups", 8))

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
        _maybe_bool(args, "--moe-router-fusion", moe.get("router_fusion", False))
        _maybe_bool(args, "--moe-layer-recompute", moe.get("layer_recompute", False))

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
        explicit_tokens = training.get("total_tokens", None)
        if explicit_tokens is not None:
            total_tokens = parse_token_count(explicit_tokens)
        else:
            total_tokens = int(training.get("tokens_per_param", 20)) * int(
                cfg.base.non_embedding_params
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
        # POET wraps per-module ColumnParallelLinear / RowParallelLinear. Grouped-GEMM
        # experts are batched weight1/weight2 Parameters (GroupedMLP / TEGroupedMLP),
        # NOT linear modules -- POET's module walk silently skips them, so every routed
        # expert would train as a dense Adam weight with zero orbit and no warning. Only
        # grouped_gemm=false (SequentialMLP, per-expert linear modules) lets POET reach
        # the experts. Fail fast rather than silently degrade.
        _moe = cfg.get("base", {}).get("model", {}).get("moe", {}) or {}
        if bool(_moe.get("enabled", False)) and bool(_moe.get("grouped_gemm", False)):
            raise ValueError(
                "[POET] optim.type=poet is incompatible with base.model.moe.grouped_gemm=true: "
                "grouped-GEMM experts are batched parameters POET cannot wrap, so they would "
                "be silently skipped (trained as dense Adam weights, no orbit). Set "
                "base.model.moe.grouped_gemm=false (SequentialMLP) to POET-ise the experts."
            )
        if poet.get("head_aligned_attn", False) and not bool(
            cfg.get("base", {}).get("model", {}).get("unfuse_qkv", False)
        ):
            raise ValueError(
                "optim.poet.head_aligned_attn requires base.model.unfuse_qkv=true "
                "(head-aligned blocks need unfused q/k/v)."
            )
        reinit_period = int(poet.get("reinit_period", 0))
        merge_period = int(poet.merge_period)
        if reinit_period > 0 and merge_period > 0 and reinit_period % merge_period != 0:
            raise ValueError(
                f"poet.reinit_period ({reinit_period}) must be a multiple of "
                f"poet.merge_period ({merge_period}); reinit boundaries must land "
                "on a folding step."
            )
        if poet.get("single_step_fast", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_fast requires merge_period=1 "
                    f"(R=Identity at forward only holds when oft_R is folded+zeroed "
                    f"every step); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_fast requires parameterization=cayley "
                    "(the factor-2 closed-form grad is the Cayley Jacobian at 0)."
                )
        if poet.get("single_step_native", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_native requires merge_period=1 "
                    f"(R=Identity at forward); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError("optim.poet.single_step_native requires parameterization=cayley.")
        if poet.get("single_step_x", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_x requires merge_period=1 "
                    f"(R=Identity at forward); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError("optim.poet.single_step_x requires parameterization=cayley.")
        if int(poet.get("head_resid_block_count", 1)) != 1:
            if not poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.head_resid_block_count > 1 requires head_aligned_attn=true "
                    "(it controls the residual side of the head-aligned POETX layer)."
                )
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.head_resid_block_count requires single_step_x=true "
                    "(the permuted-residual head layer is a POETX subclass)."
                )
        if poet.get("single_step_x_alternating", False):
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires single_step_x=true "
                    "(the alternating layer is a POETX subclass)."
                )
            if merge_period != 1:
                raise ValueError("optim.poet.single_step_x_alternating requires merge_period=1.")
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires parameterization=cayley."
                )
            if poet.get("q_optimizer", "adam") != "lie_ortho":
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires q_optimizer=lie_ortho."
                )
            if poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.single_step_x_alternating is incompatible with "
                    "head_aligned_attn=true (head-aligned uses a different layer)."
                )
            if not poet.get("train_output_rotation", True):
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires train_output_rotation=true "
                    "(both rotation sides must be trainable to alternate)."
                )
            if poet.get("lie_alternating", False):
                raise ValueError(
                    "optim.poet.single_step_x_alternating is mutually exclusive with "
                    "lie_alternating (pick one alternating path, not both)."
                )
        one_sided = poet.get("single_step_x_one_sided", None)
        if one_sided is not None:
            if one_sided not in ("in", "out"):
                raise ValueError(
                    f"optim.poet.single_step_x_one_sided must be 'in' or 'out', got {one_sided!r}."
                )
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires single_step_x=true "
                    "(the one-sided layer is a POETX subclass)."
                )
            if merge_period != 1:
                raise ValueError("optim.poet.single_step_x_one_sided requires merge_period=1.")
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires parameterization=cayley."
                )
            if poet.get("q_optimizer", "adam") != "lie_ortho":
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires q_optimizer=lie_ortho."
                )
            if poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is incompatible with head_aligned_attn=true."
                )
            if poet.get("single_step_x_alternating", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is mutually exclusive with "
                    "single_step_x_alternating."
                )
            if poet.get("lie_alternating", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is mutually exclusive with lie_alternating."
                )
            if poet.get("group_experts", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided does not support group_experts=true."
                )
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
            "--poet-reinit-period",
            reinit_period,
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
            "--poet-lie-b1",
            poet.get("lie_b1", 0.9),
            "--poet-lie-b2",
            poet.get("lie_b2", 0.95),
            "--poet-lie-eps",
            poet.get("lie_eps", 1.0e-8),
            "--poet-lie-v-mode",
            poet.get("lie_v_mode", "elementwise"),
            "--poet-lie-alternate-every",
            poet.get("lie_alternate_every", 1),
            "--poet-lie-rms-c",
            poet.get("lie_rms_c", 0.2),
            "--poet-lie-ortho-c",
            poet.get("lie_ortho_c", 0.01),
            "--poet-lie-ortho-method",
            poet.get("lie_ortho_method", "muon"),
            "--poet-lie-ortho-ns-steps",
            poet.get("lie_ortho_ns_steps", 5),
            "--poet-lie-ortho-angle-dim-exp",
            poet.get("lie_ortho_angle_dim_exp", 0.0),
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
        # store_true: alternating single-sided update for q_optimizer=lie_algebra (§6).
        if poet.get("lie_alternating", False):
            poet_args.append("--poet-lie-alternating")
        # store_true: enable Stage 2 RMS scaling (W-free) for q_optimizer=lie_algebra.
        if poet.get("lie_rms", False):
            poet_args.append("--poet-lie-rms")
        # store_true: first-vs-second moment for the lie_ortho optimizer.
        if poet.get("lie_ortho_use_second_moment", False):
            poet_args.append("--poet-lie-ortho-use-second-moment")
        # store_true: Muon-style Nesterov look-ahead on the lie_ortho skew direction.
        if poet.get("lie_ortho_nesterov", False):
            poet_args.append("--poet-lie-ortho-nesterov")
        if poet.get("lie_ortho_distributed", False):
            poet_args.append("--poet-lie-ortho-distributed")
        # store_true: cross-side decorrelation probe (ANALYSIS §17.6) — projects the two
        # sides' generators apart so cos(D_out,D_in)->0; for the simultaneous config.
        if poet.get("lie_ortho_decorrelate", False):
            poet_args.append("--poet-lie-ortho-decorrelate")
            poet_args.extend(
                [
                    "--poet-lie-ortho-decorrelate-mode",
                    poet.get("lie_ortho_decorrelate_mode", "in_off_out"),
                ]
            )
        # store_true: head-aligned attention rotation (requires unfused q/k/v).
        if poet.get("head_aligned_attn", False):
            poet_args.append("--poet-head-aligned-attn")
        if int(poet.get("head_resid_block_count", 1)) != 1:
            poet_args.append("--poet-head-resid-block-count")
            poet_args.append(str(int(poet.get("head_resid_block_count", 1))))
        if not poet.get("head_resid_perm", True):
            poet_args.append("--poet-no-head-resid-perm")
        # store_true: single-step (R=I) fast path (closed-form backward).
        if poet.get("single_step_fast", False):
            poet_args.append("--poet-single-step-fast")
        # store_true: gather-free native-frame single-step path.
        if poet.get("single_step_native", False):
            poet_args.append("--poet-single-step-native")
        # store_true: forward-frame perm-free-forward path.
        if poet.get("single_step_x", False):
            poet_args.append("--poet-single-step-x")
        # store_true: dedicated true-single-side alternating POETX layer.
        if poet.get("single_step_x_alternating", False):
            poet_args.append("--poet-single-step-x-alternating")
        # Pure one-sided POETX layer (train one fixed rotation side for the run).
        if poet.get("single_step_x_one_sided", None) is not None:
            poet_args.append("--poet-single-step-x-one-sided")
            poet_args.append(str(poet.get("single_step_x_one_sided")))
        # store_true: group MoE experts into a single batched POETX layer.
        if poet.get("group_experts", False):
            poet_args.append("--poet-group-experts")
        return _sequence(poet_args)

    if kind == "ngpt_adamw":
        # The nGPT *architecture* flags (--ngpt, --ngpt-*-init, --ngpt-no-warmup)
        # are emitted by _ngpt_arch_args, keyed on experiment.kind=='ngpt' rather
        # than on this optimizer type, so the nGPT architecture can be trained
        # with a different optimizer (muon_kimi, poet) by swapping optim.type
        # while keeping experiment.kind=='ngpt'. This branch emits only the
        # AdamW optimizer flags.
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--slm-optimizer",
                "ngpt_adamw",
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
            ]
        )

    raise ValueError(f"Unsupported optimizer type {kind!r}")


def _ngpt_arch_args(cfg: DictConfig) -> list[str]:
    """nGPT *architecture* CLI flags, decoupled from the optimizer type.

    The nGPT hypersphere architecture is requested via ``experiment.kind ==
    'ngpt'``, NOT via ``optim.type``. Emitting ``--ngpt`` and the scaling-vector
    inits here -- instead of inside the ``ngpt_adamw`` optimizer branch -- lets
    the nGPT architecture be combined with a non-nGPT optimizer (``muon_kimi``,
    ``poet``, ...): the optimizer branch supplies the optimizer flags and this
    helper supplies the architecture flags. Without this split, swapping
    ``optim.type`` away from ``ngpt_adamw`` would silently drop ``--ngpt`` and
    train a plain (non-nGPT) model.

    The scaling-vector init values live under ``optim.ngpt.*`` (architecture
    hyperparameters that ride along with the optimizer block). When the nGPT
    architecture is not requested this is a no-op (returns ``[]``).
    """
    experiment = cfg.get("experiment", {}) or {}
    if str(experiment.get("kind", "")) != "ngpt":
        return []
    ng = (cfg.get("optim", {}) or {}).get("ngpt", {}) or {}
    arch = [
        "--ngpt",
        "--ngpt-alpha-init",
        float(ng.get("alpha_init", 0.05)),
        "--ngpt-sqk-init",
        float(ng.get("sqk_init", 1.0)),
        "--ngpt-suv-init",
        float(ng.get("suv_init", 1.0)),
        "--ngpt-sz-init",
        float(ng.get("sz_init", 1.0)),
    ]
    if bool(ng.get("no_warmup", True)):
        arch.append("--ngpt-no-warmup")
    return _sequence(arch)


def _pgpt_arch_args(cfg: DictConfig) -> list[str]:
    """pgpt architecture CLI flags (mirror of _ngpt_arch_args, keyed on kind=='pgpt').

    pgpt reuses nGPT's architecture flags and config fields (the forked pgpt model
    reads config.ngpt_* exactly like nGPT). Emitting them here, keyed on
    experiment.kind=='pgpt' (NOT optim.type, which is 'poet'), keeps the optimizer
    branch free to supply the POET flags. No-op when pgpt is not requested.
    """
    experiment = cfg.get("experiment", {}) or {}
    if str(experiment.get("kind", "")) != "pgpt":
        return []
    ng = (cfg.get("optim", {}) or {}).get("ngpt", {}) or {}
    arch = [
        "--ngpt",
        "--ngpt-alpha-init",
        float(ng.get("alpha_init", 0.05)),
        "--ngpt-sqk-init",
        float(ng.get("sqk_init", 1.0)),
        "--ngpt-suv-init",
        float(ng.get("suv_init", 1.0)),
        "--ngpt-sz-init",
        float(ng.get("sz_init", 1.0)),
    ]
    if bool(ng.get("no_warmup", True)):
        arch.append("--ngpt-no-warmup")
    return _sequence(arch)


def _distributed_optimizer_supported(cfg: DictConfig) -> bool:
    """Whether Megatron's sharded distributed optimizer can drive this optimizer.

    - ``adamw``: yes (stock Megatron path).
    - ``poet``: no, on EVERY path. ``get_megatron_poet_optimizer``
      (``src/optim/poet.py``) raises ``"POET optimizer does not support
      distributed optimizer"`` before it builds anything -- the guard precedes the
      stock-Megatron-Adam (``_build_vanilla_poet_optimizer``), custom-POETAdam,
      and Lie/Muon-on-Q builders alike. Sharding POET would require the
      ``poet_merge_step`` fold + Adam-momentum reset to be sharded-master aware,
      which it is not. So the launcher must never request one for POET: otherwise
      the flag is emitted but the optimizer build hard-crashes at first step.
      (This previously returned True for the stock-Adam path on the belief that
      the vanilla builder honors ``config.use_distributed_optimizer``; the
      unconditional guard makes that unreachable, and a real run crashes.)
    - everything else (``muon``, ``ngpt``, ...): no.
    """
    # Only stock AdamW drives Megatron's sharded distributed optimizer; POET
    # (and muon/ngpt) build/route their own optimizer and reject it.
    return cfg.optim.type == "adamw"


def _parallel_args(cfg: DictConfig) -> list[str]:
    parallelism = cfg.parallelism
    args: list[str] = []
    _add(args, "--tensor-model-parallel-size", parallelism.get("tp", 1))
    _add(args, "--pipeline-model-parallel-size", parallelism.get("pp", 1))
    if bool(parallelism.get("sequence_parallel", True)):
        _add(args, "--sequence-parallel")
    if bool(parallelism.get("distributed_optimizer", False)) and _distributed_optimizer_supported(
        cfg
    ):
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


def _weight_norm_args(training: DictConfig) -> list[str]:
    """Emit the weight_norm_monitor flags from the `training` config block."""
    args: list[str] = []
    _maybe_bool(args, "--log-weight-norms", training.get("log_weight_norms", False))
    interval = training.get("log_weight_norms_interval", None)
    if interval is not None:
        _add(args, "--log-weight-norms-interval", interval)
    layers = training.get("weight_norm_layers", None)
    if layers is not None:
        _add(args, "--weight-norm-layers", layers)
    return args


def _delta_w_args(training: DictConfig) -> list[str]:
    """Emit the weight_delta_monitor flags from the `training` config block."""
    if not _truthy(training.get("log_delta_w", False)):
        return []
    args: list[str] = ["--log-delta-w"]
    interval = training.get("log_delta_w_interval", None)
    if interval is not None:
        _add(args, "--log-delta-w-interval", interval)
    layers = training.get("delta_w_layers", None)
    if layers is not None:
        _add(args, "--delta-w-layers", layers)
    max_targets = training.get("delta_w_max_targets", None)
    if max_targets is not None:
        _add(args, "--delta-w-max-targets", max_targets)
    spectral_max_dim = training.get("delta_w_spectral_max_dim", None)
    if spectral_max_dim is not None:
        _add(args, "--delta-w-spectral-max-dim", spectral_max_dim)
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
    else:
        # With --save omitted, Megatron's wandb writer would deref
        # os.path.join(args.save, 'wandb') on a None save dir and crash. Give
        # wandb its own dir so logging works without writing checkpoints.
        _add(args, "--wandb-save-dir", f"{archive}/wandb")
    # Only force an entity when one is configured. An empty/null entity lets
    # wandb fall back to the logged-in account's default (personal) namespace,
    # avoiding "entity ... not found" when the configured team is inaccessible.
    entity = cfg.wandb.get("entity", None)
    if entity:
        _add(args, "--wandb-entity", str(entity))
    if resume:
        _add(args, "--finetune")
        _add(args, "--override-opt-param-scheduler")
    args.extend(_weight_norm_args(cfg.training))
    args.extend(_delta_w_args(cfg.training))
    return args


def build_megatron_args(cfg: DictConfig) -> list[str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    args: list[str] = []
    args.extend(_model_args(cfg))
    args.extend(_training_args(cfg))
    args.extend(_optimizer_args(cfg))
    args.extend(_ngpt_arch_args(cfg))
    args.extend(_pgpt_arch_args(cfg))
    args.extend(_parallel_args(cfg))
    args.extend(_data_args(cfg))
    args.extend(_logging_args(cfg))
    return args
