"""Translate a resolved slm-research config into a torchtitan JobConfig.

Returns ``(toml_dict, overrides)``:
  * ``toml_dict``  -> serialized to ``<run_dir>/torchtitan.toml`` and passed as
                      ``--job.config_file``.
  * ``overrides``  -> dotted ``--section.key value`` CLI args appended after it.

Pure function: no torch, no torchtitan import. The TOML *key names* below are the
ones recorded in docs/torchtitan_api_notes.md §1 for the v0.2.2 pin; if a bump
moves them, update both files together.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

# slm family -> torchtitan native model name (verified to ship at v0.2.2).
_FAMILY_TO_TITAN = {"llama3": "llama3", "qwen3": "qwen3", "deepseek_v3": "deepseek_v3"}
# Families where the shared dense/GQA dims map cleanly onto torchtitan's args, so
# we register a custom `slm_<scale>` flavor. deepseek_v3 is EXCLUDED: its args
# (MLA ranks + MoE sizing: inter_dim/moe_inter_dim/n_dense_layers/q_lora_rank/...)
# don't follow from slm's dense dims, so M1 uses a NATIVE deepseek flavor as-is.
_SLM_FLAVOR_FAMILIES = {"llama3", "qwen3"}


def _adam_lr(optim: DictConfig) -> float:
    if optim.get("lr", None) is not None:
        return float(optim.lr)
    return float(optim.get("adam", {}).get("lr", 1.0e-3))


def _model_block(cfg: DictConfig) -> dict:
    # torchtitan's [model] TOML carries ONLY name + flavor (+ asset paths). The
    # model DIMENSIONS live in the registered model_args flavor, NOT in TOML, so
    # src/titan_ext clones torchtitan's NATIVE model of this family and (for the
    # dense families) adds an `slm_<scale>` flavor from SLM_RESOLVED_CONFIG.
    family = str(cfg.base.family)
    if family not in _FAMILY_TO_TITAN:
        raise ValueError(
            f"torchtitan backend supports families {sorted(_FAMILY_TO_TITAN)}; got {family!r}"
        )
    if family in _SLM_FLAVOR_FAMILIES:
        flavor = f"slm_{cfg.base.scale}"  # slm-registered model_args (size)
    else:
        # deepseek_v3 M1: pick a torchtitan-native flavor (overridable per scale
        # via base.model.titan_flavor); a full deepseek dims-mapper is a follow-on.
        flavor = str(cfg.base.model.get("titan_flavor", "debugmodel"))
    block = {
        "name": f"slm_{family}",  # slm-registered clone of torchtitan's native model
        "flavor": flavor,
    }
    # torchtitan builds a HF tokenizer at Trainer.__init__ from model.hf_assets_path
    # (v0.2.2: HuggingFaceTokenizer loads <path>/tokenizer.json; the old
    # ./assets/tokenizer default was deprecated in PR #1540, so leaving it unset
    # crashes the run). Point it at the SAME HF tokenizer dir the slm/Megatron data
    # pipeline used to pre-tokenize this corpus (cfg.data.tokenizer_model) so vocab
    # and special tokens match the Megatron-indexed data.
    tokenizer_dir = cfg.data.get("tokenizer_model", None)
    if not tokenizer_dir:
        raise ValueError(
            "torchtitan needs a HF tokenizer directory: set data.tokenizer_model "
            "(forwarded to torchtitan's model.hf_assets_path)."
        )
    block["hf_assets_path"] = str(tokenizer_dir)
    return block


def _training_block(cfg: DictConfig) -> dict:
    seq_len = int(cfg.base.model.seq_length)
    gbs = int(cfg.training.global_batch_size)
    # Honor an explicit training.steps (e.g. `training.steps=20` smoke runs); else
    # derive the optimizer-step count from the token budget. steps feeds BOTH the LR
    # schedule AND the dataloader's num_samples (= steps * global_batch_size). The
    # full schedule consumes (total_tokens // seq_len) SAMPLES — exactly the Megatron
    # path's --train-samples — so the optimizer-step count is that divided by the
    # global batch size. Dividing by seq_len alone (the previous formula) overcounted
    # steps, the LR-schedule length, AND the built dataset index by global_batch_size.
    steps_override = cfg.training.get("steps", None)
    if steps_override:
        steps = int(steps_override)
    else:
        steps = int(cfg.training.total_tokens) // (seq_len * gbs)
    return {
        "seq_len": seq_len,
        "global_batch_size": gbs,
        "steps": steps,
        "mixed_precision_param": "bfloat16",  # M1 baseline; Float8 is a follow-on
        "max_norm": float(cfg.training.get("clip_grad", 1.0) or 1.0),
        # Coordinates the slm_megatron_indexed dataloader (Task 9) reads; the data
        # path/seed are also re-read from SLM_RESOLVED_CONFIG at train time.
        "dataset": "slm_megatron_indexed",
        "dataset_path": str(cfg.data.path),
    }


def _optimizer_block(cfg: DictConfig) -> dict:
    optim = cfg.optim
    if str(optim.type) != "adamw":
        raise ValueError(
            f"torchtitan backend only supports adamw in milestone 1; got {optim.type!r}"
        )
    betas = list(optim.get("betas", [0.9, 0.95]))
    return {
        "name": "AdamW",
        "lr": _adam_lr(optim),
        "eps": float(optim.get("eps", 1.0e-8)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "weight_decay": float(optim.get("weight_decay", 0.1)),
    }


def _parallelism_block(cfg: DictConfig) -> dict:
    par = cfg.parallelism
    return {
        "tensor_parallel_degree": int(par.get("tp", 1)),
        "pipeline_parallel_degree": int(par.get("pp", 1)),
        "context_parallel_degree": 1,
        # -1 => FSDP2 shards over all remaining (world / TP / PP / CP) ranks.
        "data_parallel_shard_degree": -1,
        "data_parallel_replicate_degree": 1,
    }


def _metrics_block(cfg: DictConfig) -> dict:
    return {
        "enable_wandb": not bool(cfg.cluster.get("wandb_offline", False)),
    }


def _comm_block(cfg: DictConfig) -> dict:
    # The slm_megatron_indexed dataloader builds its per-run sample/shuffle index on
    # rank 0 while the other ranks wait at a torch.distributed.barrier(). A COLD build
    # of a large corpus (tens of millions of samples) takes ~20-30 min — far longer
    # than torchtitan's default comm.init_timeout_seconds=300 (5 min), so the waiting
    # ranks hit a barrier timeout and crash the run before the first step. The
    # torchtitan path always cold-builds (its seed + FixedVocabTokenizer give a
    # different cache hash than the Megatron path), so this bites on every fresh
    # data shape. Give the init/first-step phase a generous budget; warm cache loads
    # are fast and unaffected. Overridable via cluster.comm_init_timeout_seconds.
    return {
        "init_timeout_seconds": int(cfg.cluster.get("comm_init_timeout_seconds", 3600)),
    }


def lr_scheduler_block(sched: dict, *, total_steps: int) -> dict:
    """Map an slm `scheduler` block to torchtitan [lr_scheduler] keys.

    torchtitan uses `decay_ratio` (a FRACTION of total steps), `decay_type`, and
    `min_lr_factor` — there is NO `decay_steps`. slm's `wsd_decay_fraction` is
    already a ratio, so it maps straight to `decay_ratio`.
    """
    # slm wsd_decay_style -> torchtitan decay_type (names verified in api notes §1).
    decay_type_map = {
        "cosine": "cosine",
        "linear": "linear",
        "minus_sqrt": "sqrt",
        "exponential": "linear",
    }
    warmup_frac = float(sched.get("warmup_fraction", 0.0) or 0.0)
    block = {
        "warmup_steps": int(round(warmup_frac * total_steps)),
        "min_lr_factor": float(sched.get("min_lr_ratio", 0.0) or 0.0),
    }
    if str(sched.get("type", "")).lower() == "wsd":
        block["decay_ratio"] = float(sched.get("wsd_decay_fraction", 0.0) or 0.0)
        block["decay_type"] = decay_type_map.get(
            str(sched.get("wsd_decay_style", "cosine")), "cosine"
        )
    return block


def unmapped_megatron_knobs(cfg: DictConfig) -> list[str]:
    """Human-readable notes for Megatron-only signals torchtitan ignores."""
    notes: list[str] = []
    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    if patches:
        notes.append(
            f"experiment.patches {patches} — Megatron monkey-patches, ignored on torchtitan"
        )
    if bool(cfg.base.model.get("use_sandwich_norm", False)):
        notes.append("base.model.use_sandwich_norm — no torchtitan-native equivalent")
    return notes


def build_torchtitan_config(cfg: DictConfig) -> tuple[dict, list[str]]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    toml: dict = {
        "model": _model_block(cfg),
        "training": _training_block(cfg),
        "optimizer": _optimizer_block(cfg),
        "parallelism": _parallelism_block(cfg),
        "metrics": _metrics_block(cfg),
        # Raise the init/first-step barrier timeout so rank 0's cold dataset-index
        # build doesn't trip torchtitan's 300s default (see _comm_block).
        "comm": _comm_block(cfg),
        # VERIFIED v0.2.2: seed is [debug].seed, not [training].seed
        # (docs/torchtitan_api_notes.md §1).
        "debug": {"seed": int(cfg.seed)},
    }
    toml["lr_scheduler"] = lr_scheduler_block(
        OmegaConf.to_container(cfg.scheduler, resolve=True), total_steps=toml["training"]["steps"]
    )
    overrides: list[str] = []  # dataloader (Task 9) appends here
    return toml, overrides
