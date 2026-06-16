"""Patch: swap GPT layer spec to nGPT, stamp config fields, attach sz.

Targets (all triggered only when args.ngpt is set):

- `megatron.training.arguments.core_transformer_config_from_args` —
  wrap to copy nGPT-specific args onto the returned TransformerConfig
  (`softmax_scale`, `ngpt_base_scale`, `ngpt_alpha_init`,
  `ngpt_sqk_init`, `ngpt_suv_init`). The softmax_scale stamp is what
  flips Megatron's `DotProductAttention` from `1/sqrt(head_dim)` to
  `sqrt(head_dim)`. We wrap this BEFORE importing gpt_builders so the
  `from megatron.training.arguments import ...` inside gpt_builders binds
  the wrapped version (patches are applied before pretrain_gpt import).

- `gpt_builders.gpt_builder` — wrap so `_get_transformer_layer_spec`
  returns our nGPT spec. After model construction: attach `sz`,
  register the weight-normalization role map, and run the one-shot
  initial L2 projection that the reference does at train.py:411.

Upstream SHA: see docs/megatron_pin.md.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

from src.patches._registry import register_patch


def _rebind_if_stale(module_name: str, attr: str, orig, wrapped) -> None:
    """Rebind a stale by-value import of a function we just wrapped.

    ``ngpt_apply_spec`` wraps functions on one module that consumers captured
    by value (``from X import Y``) on another: ``gpt_builder`` (consumed as
    ``pretrain_gpt.gpt_builder`` by the launcher) and
    ``core_transformer_config_from_args`` (consumed bare inside
    ``gpt_builders``). If a patch sorting before us (``model_unfuse_linears``)
    imported those modules first, the copies are frozen to the originals.
    Rebind ``module_name.attr`` to ``wrapped`` — but only if it still holds the
    ``orig`` we wrapped, so we never clobber a different wrapper. No-op (and no
    import) if the module is not yet loaded; in that ordering the consumer
    imports after us and binds the wrapped function naturally.
    """
    m = sys.modules.get(module_name)
    if m is not None and getattr(m, attr, None) is orig:
        setattr(m, attr, wrapped)


_TARGET = (
    "gpt_builders.gpt_builder",
    "megatron.training.arguments.core_transformer_config_from_args",
)
logger = logging.getLogger(__name__)


@register_patch(name="ngpt_apply_spec", targets=_TARGET)
def apply() -> None:
    # ---- Wrap config builder ----
    from megatron.training import arguments as _ma

    _orig_cfg = _ma.core_transformer_config_from_args

    def _wrapped_cfg(args, *a, **kw):
        config = _orig_cfg(args, *a, **kw)
        if not getattr(args, "ngpt", False):
            return config
        # softmax_scale: nGPT uses sqrt(head_dim) instead of 1/sqrt(head_dim).
        head_dim = int(args.hidden_size) // int(args.num_attention_heads)
        config.softmax_scale = math.sqrt(head_dim)
        # ngpt_* fields read by NGPTTransformerLayer.__init__ and the spec.
        hidden = int(args.hidden_size)
        config.ngpt_base_scale = float(
            getattr(args, "ngpt_base_scale", None) or (1.0 / math.sqrt(hidden))
        )
        config.ngpt_alpha_init = float(getattr(args, "ngpt_alpha_init", 0.05))
        config.ngpt_sqk_init = float(getattr(args, "ngpt_sqk_init", 1.0))
        config.ngpt_suv_init = float(getattr(args, "ngpt_suv_init", 1.0))
        # NGPTMLP reads this to build split u/v projections (parity with the
        # unfused baselines). Attention unfuse is handled by model_unfuse_linears.
        config.unfuse_fc1 = bool(getattr(args, "unfuse_fc1", False))
        config.ngpt = True  # boolean shortcut for downstream checks
        return config

    _ma.core_transformer_config_from_args = _wrapped_cfg
    _rebind_if_stale("gpt_builders", "core_transformer_config_from_args", _orig_cfg, _wrapped_cfg)

    # ---- Wrap GPT model builder ----
    import gpt_builders as _gb  # third_party/Megatron-LM is on sys.path

    from src.model.ngpt.output_scaling import attach_sz_scaling
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    _orig_builder = _gb.gpt_builder

    def _wrapped_builder(args, *a, **kw):
        if not getattr(args, "ngpt", False):
            return _orig_builder(args, *a, **kw)
        from megatron.core.transformer.transformer_config import TransformerConfig

        original_get_spec = _gb._get_transformer_layer_spec

        def _ngpt_get_spec(use_te: bool, config: TransformerConfig):
            return build_ngpt_layer_spec(config)

        _gb._get_transformer_layer_spec = _ngpt_get_spec
        try:
            model = _orig_builder(args, *a, **kw)
        finally:
            _gb._get_transformer_layer_spec = original_get_spec

        # Post-build hooks: sz scaling + weight-norm role registration.
        chunks = model if isinstance(model, list) else [model]
        for m in chunks:
            attach_sz_scaling(
                m,
                vocab_size=args.padded_vocab_size,
                base_scale=float(
                    getattr(args, "ngpt_base_scale", None) or (1.0 / math.sqrt(args.hidden_size))
                ),
            )
            _register_ngpt_norm_roles(m, expected_layers=int(args.num_layers))
            _normalize_now(m)  # one-shot init normalize (reference train.py:411)
        logger.info("[nGPT] applied spec + attached sz + registered weight-norm roles")
        return model

    _gb.gpt_builder = _wrapped_builder
    _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig_builder, _wrapped_builder)


# ---------------------------------------------------------------------------
# Weight-normalization role map
# ---------------------------------------------------------------------------
#
# `_NORM_ROLES_BY_SUFFIX` maps the *trailing* qualified-name segments
# (after the last dot) of every nGPT weight matrix to its normalization
# role. Trailing-segment matching is intentional: a raw substring match
# (`"v_proj" in "qkv_proj.oft_R"` -> True) silently drops/double-counts.

_NORM_ROLES_BY_SUFFIX: dict[tuple[str, ...], str] = {
    # Embedding row = per-token vector -> unit norm along hidden.
    ("embedding", "word_embeddings", "weight"): "rows",
    # LM head row = per-vocab vector -> unit norm along hidden.
    ("output_layer", "weight"): "rows",
    # Q/K/V projection rows = per-output-channel vectors (fused).
    ("linear_qkv", "weight"): "rows",
    # ...and the unfused split (model_unfuse_linears splits linear_qkv).
    ("linear_q", "weight"): "rows",
    ("linear_k", "weight"): "rows",
    ("linear_v", "weight"): "rows",
    # Attention output projection columns = per-input-channel vectors.
    ("linear_proj", "weight"): "cols",
    # SwiGLU c_fc rows = per-output-channel vectors (fused gate+up concat).
    ("linear_fc1", "weight"): "rows",
    # ...and nGPT's unfused MLP split: NGPTMLP builds linear_fc1_u/v when
    # unfuse_fc1 is set (each row is still a per-output-channel vector).
    ("linear_fc1_u", "weight"): "rows",
    ("linear_fc1_v", "weight"): "rows",
    # SwiGLU mlp_c_proj columns = per-input-channel vectors.
    ("linear_fc2", "weight"): "cols",
}


def _match_role(name: str) -> str | None:
    parts = name.split(".")
    for suffix, role in _NORM_ROLES_BY_SUFFIX.items():
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == suffix:
            return role
    return None


def _register_ngpt_norm_roles(model, expected_layers: int) -> None:
    """Build a {param -> 'rows'|'cols'} dict on `model._ngpt_norm_role_map`.

    Tied embeddings make `output_layer.weight` and
    `embedding.word_embeddings.weight` alias the same parameter; both
    roles agree ("rows") so the dict-overwrite is benign.
    """
    role_map: dict[Any, str] = {}
    matched_per_role = {"rows": 0, "cols": 0}
    for name, param in model.named_parameters():
        role = _match_role(name)
        if role is None:
            continue
        role_map[param] = role
        matched_per_role[role] += 1

    # Sanity check: detect future Megatron renames or missed weight matrices.
    # Per-layer matrices (fused case): linear_qkv (rows), linear_proj (cols),
    # linear_fc1 (rows), linear_fc2 (cols) = 4 per layer. Unfusing splits
    # qkv -> q/k/v and fc1 -> fc1_u/v, giving MORE matches, so the >=4-per-layer
    # floor below still holds. Plus embedding (rows) and output_layer (rows) —
    # counted once under tying because Megatron's `named_parameters` dedups.
    n_unique_params = len(role_map)
    per_layer = 4
    embedding_plus_head_min = 1  # at least the embedding; output_layer aliases it under tying
    expected_min = expected_layers * per_layer + embedding_plus_head_min
    assert n_unique_params >= expected_min, (
        f"nGPT weight-norm role map matched only {n_unique_params} params; "
        f"expected >= {expected_min} (got rows={matched_per_role['rows']} "
        f"cols={matched_per_role['cols']}). A param-name regression in "
        "Megatron would slip through silently — fix _NORM_ROLES_BY_SUFFIX."
    )
    model._ngpt_norm_role_map = role_map


def _normalize_now(model) -> None:
    from src.model.ngpt.normalize import normalize_module_matrices

    if hasattr(model, "_ngpt_norm_role_map"):
        normalize_module_matrices(model._ngpt_norm_role_map)
