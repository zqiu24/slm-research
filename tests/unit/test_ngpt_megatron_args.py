"""Tests that the nGPT experiment YAML composes into the right Megatron CLI args."""

from omegaconf import OmegaConf

from src.utils.megatron_args import build_megatron_args


def _ngpt_cfg():
    return OmegaConf.create(
        {
            "base": {
                "family": "llama3",
                "scale": "600m",
                "non_embedding_params": 600_000_000,
                "model": {
                    "num_layers": 4,
                    "hidden_size": 32,
                    "ffn_hidden_size": 128,
                    "num_attention_heads": 4,
                    "num_query_groups": 4,
                    "head_dim": 8,
                    "seq_length": 64,
                    "max_position_embeddings": 64,
                    "positional_encoding": "rope",
                    "rotary_base": 500000,
                    "attention_dropout": 0.0,
                    "hidden_dropout": 0.0,
                    "normalization": "RMSNorm",
                    "norm_epsilon": 1e-5,
                    "init_method_std": 0.02,
                    "tie_embeddings": True,
                    "attention_backend": "flash",
                    "qk_norm": False,
                },
            },
            "training": {
                "tokens_per_param": 20,
                "global_batch_size": 64,
                "seq_length": 64,
                "micro_batch_size": 1,
                "log_interval": 1,
                "eval_iters": 1,
                "eval_interval": 1,
                "save_interval": 100000,
            },
            "optim": {
                "type": "ngpt_adamw",
                "lr": 15e-4,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "ngpt": {
                    "alpha_init": 0.05,
                    "sqk_init": 1.0,
                    "suv_init": 1.0,
                    "sz_init": 1.0,
                    "no_warmup": True,
                },
            },
            "scheduler": {
                "type": "cosine",
                "warmup_fraction": 0.01,
                "min_lr_ratio": 0.1,
            },
            "parallelism": {"tp": 1, "pp": 1, "sequence_parallel": False},
            "data": {
                "path": "/tmp/x",
                "tokenizer_type": "GPT2BPETokenizer",
                "tokenizer_model": "/tmp/t",
                "vocab_size": 100,
                "name": "x",
                "split": "100,0,0",
                "no_mmap_bin_files": False,
                "no_create_attention_mask_in_dataloader": False,
                "num_workers": 0,
            },
            "wandb": {"project": "test"},
            "experiment": {"name": "ngpt", "kind": "ngpt"},
            "seed": 0,
        }
    )


def test_emits_ngpt_flag():
    args = build_megatron_args(_ngpt_cfg())
    assert "--ngpt" in args


def test_emits_ngpt_init_flags():
    args = build_megatron_args(_ngpt_cfg())
    for flag in ("--ngpt-alpha-init", "--ngpt-sqk-init", "--ngpt-suv-init", "--ngpt-sz-init"):
        assert flag in args, f"missing flag: {flag}"


def test_no_warmup_emits_zero_warmup_samples():
    args = build_megatron_args(_ngpt_cfg())
    i = args.index("--lr-warmup-samples")
    assert int(args[i + 1]) == 0


def test_no_warmup_false_keeps_default_warmup():
    cfg = _ngpt_cfg()
    cfg.optim.ngpt.no_warmup = False
    args = build_megatron_args(cfg)
    # With no_warmup False the scheduler's warmup applies; the cosine default
    # expresses warmup as a fraction of the budget, not an explicit sample count.
    i = args.index("--lr-warmup-fraction")
    assert float(args[i + 1]) > 0
    assert "--lr-warmup-samples" not in args


def test_disable_bias_linear_still_present():
    """nGPT relies on disable-bias-linear (default in _model_args). Sanity check."""
    args = build_megatron_args(_ngpt_cfg())
    assert "--disable-bias-linear" in args


def _muon_optim():
    """muon_kimi optimizer block (mirrors configs/experiments/optim/muon_kimi.yaml),
    plus the nGPT scaling-vector init sub-block that rides along with the arch."""
    return OmegaConf.create(
        {
            "type": "muon_kimi",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "muon_momentum": 0.95,
            "muon_use_nesterov": True,
            "muon_num_ns_steps": 5,
            "adam": {"betas": [0.9, 0.95], "eps": 1e-8},
            "ngpt": {
                "alpha_init": 0.05,
                "sqk_init": 1.0,
                "suv_init": 1.0,
                "sz_init": 1.0,
                "no_warmup": True,
            },
        }
    )


def test_ngpt_arch_emitted_with_non_ngpt_optimizer():
    """The nGPT architecture (--ngpt) is keyed on experiment.kind=='ngpt', NOT on
    optim.type: selecting a muon optimizer must keep the hypersphere arch flags
    while emitting the muon optimizer (not ngpt_adamw)."""
    cfg = _ngpt_cfg()
    cfg.optim = _muon_optim()
    args = build_megatron_args(cfg)
    # Architecture flags still present.
    assert "--ngpt" in args
    for flag in ("--ngpt-alpha-init", "--ngpt-sqk-init", "--ngpt-suv-init", "--ngpt-sz-init"):
        assert flag in args, f"missing arch flag: {flag}"
    # Optimizer is muon_kimi, not ngpt_adamw.
    assert "muon_kimi" in args
    assert "ngpt_adamw" not in args


def test_ngpt_flag_absent_without_ngpt_kind():
    """A plain (non-nGPT) experiment must NOT emit --ngpt even with a muon
    optimizer; the architecture is gated solely on experiment.kind."""
    cfg = _ngpt_cfg()
    cfg.experiment = OmegaConf.create({"name": "muon_kimi", "family": "optim"})
    cfg.optim = _muon_optim()
    args = build_megatron_args(cfg)
    assert "--ngpt" not in args
    assert "--ngpt-alpha-init" not in args
