"""slm-research runtime extensions for torchtitan.

Imported via torchtitan's experimental.custom_import hook BEFORE TrainSpec
lookup (see launchers/train_torchtitan.py CUSTOM_IMPORT_FLAG). On import it reads
the resolved slm config (SLM_RESOLVED_CONFIG), clones torchtitan's NATIVE spec
for the configured family, adds an `slm_<scale>` flavor, and registers the whole
thing as a new TrainSpec "slm_<family>". Task 9 also swaps in the Megatron-indexed
dataloader.

Importing this package MUST NOT raise when torchtitan is absent (CPU unit-test
env) — registration is guarded. It must also be idempotent: register_train_spec
RAISES on a duplicate name, so we skip if "slm_<family>" already exists.
"""

from __future__ import annotations

import dataclasses
import os

# Keep in sync with src/utils/torchtitan_args (_FAMILY_TO_TITAN, _SLM_FLAVOR_FAMILIES).
_FAMILY_TO_TITAN = {"llama3": "llama3", "qwen3": "qwen3", "deepseek_v3": "deepseek_v3"}
_SLM_FLAVOR_FAMILIES = {"llama3", "qwen3"}  # deepseek_v3 uses a native flavor in M1


def _dims_from(cfg) -> dict:
    m = cfg.base.model
    # Superset of dense/GQA dim fields; build_slm_flavor's hasattr filter keeps
    # only the ones the target family's args class actually has (e.g. llama3 has
    # no `hidden_dim`/`head_dim`; qwen3 has both — so qwen3 even honors slm's FFN).
    return {
        "dim": int(m.hidden_size),
        "n_layers": int(m.num_layers),
        "n_heads": int(m.num_attention_heads),
        "n_kv_heads": int(m.num_query_groups),
        "head_dim": int(m.head_dim),
        "hidden_dim": int(m.ffn_hidden_size),  # qwen3 explicit FFN; llama3/deepseek lack this field
        "vocab_size": int(cfg.base.tokenizer.nominal_vocab_size),
        "rope_theta": float(m.rotary_base),
        "norm_eps": float(m.norm_epsilon),
        "max_seq_len": int(m.seq_length),
    }


def _slm_spec_from(base, cfg):
    """Clone native `base`; for dense families add an slm_<scale> flavor sized
    from `cfg`. deepseek_v3 keeps `base`'s native flavors unchanged (its MLA/MoE
    args don't map from slm's dense dims — see _SLM_FLAVOR_FAMILIES).

    Native LR scheduler / optimizer / parallelize fns are inherited from `base`.
    Task 9 additionally sets build_dataloader_fn here. TrainSpec field names are
    the verified ones (docs/torchtitan_api_notes.md §2).
    """
    from src.titan_ext.dataloader import build_dataloader as _dataloader
    from src.titan_ext.model_flavor import build_slm_flavor, pick_template

    model_args = dict(base.model_args)  # copy native flavor mapping
    if str(cfg.base.family) in _SLM_FLAVOR_FAMILIES:
        template = pick_template(model_args)
        model_args[f"slm_{cfg.base.scale}"] = build_slm_flavor(template, _dims_from(cfg))
    # Swap torchtitan's stock (C4) dataloader for the Megatron-indexed one so the
    # torchtitan backend reads the same .bin/.idx corpus (api notes §2/§4, M2).
    return dataclasses.replace(base, model_args=model_args, build_dataloader_fn=_dataloader)


def _register() -> None:
    try:
        from torchtitan.protocols.train_spec import (  # api notes §2
            get_train_spec,
            register_train_spec,
        )
    except Exception:
        return  # torchtitan not importable (CPU unit-test env): no-op
    if "SLM_RESOLVED_CONFIG" not in os.environ:
        return  # no config available yet (e.g. import before launch sets the env)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.environ["SLM_RESOLVED_CONFIG"])
    titan = _FAMILY_TO_TITAN.get(str(cfg.base.family))
    if titan is None:
        return  # unsupported family on torchtitan: leave registry untouched
    slm_name = f"slm_{cfg.base.family}"
    try:
        get_train_spec(slm_name)
        return  # already registered (idempotent — register_train_spec raises on dup)
    except Exception:
        pass
    base = get_train_spec(titan)
    register_train_spec(slm_name, _slm_spec_from(base, cfg))


def _patch_metrics() -> None:
    # Rank-0-only per-step console line + ETA, and canonical W&B metric keys, to
    # match / align with the Megatron path. Independent of the TrainSpec
    # registration above and of SLM_RESOLVED_CONFIG; each no-ops if torchtitan is
    # absent (CPU unit-test env).
    from src.titan_ext.metrics import (
        apply_titan_metrics_patch,
        apply_titan_wandb_normalize,
    )

    apply_titan_metrics_patch()
    apply_titan_wandb_normalize()


def _patch_validation() -> None:
    # Point torchtitan's Validator at the Megatron-indexed validation split so
    # [validation].enable evals on the same corpus as training (comparable to the
    # Megatron backend's eval). Must run before Trainer init builds the Validator;
    # titan_ext is imported via experimental.custom_import at startup. No-ops if
    # torchtitan is absent (CPU unit-test env).
    from src.titan_ext.dataloader import (
        apply_titan_validation_dataloader_patch,
        apply_titan_validation_schedule_patch,
    )

    apply_titan_validation_dataloader_patch()
    apply_titan_validation_schedule_patch()  # no step-1 eval (avoid a distorted first point)


_register()
_patch_metrics()
_patch_validation()
