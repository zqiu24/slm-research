# src/patches/weight_norm_monitor.py
"""Patch: log row/column norm summaries of a few weight matrices to W&B.

Flag-gated by ``--log-weight-norms`` (interval ``--log-weight-norms-interval``,
default 100; layers ``--weight-norm-layers``, default ``first,mid,last``). Inert
otherwise, so it is safe in ``_ALWAYS_ON_PATCHES``.

Mechanism: wrap ``train_step`` as the OUTER wrapper. ``weight_norm_monitor``
sorts after ``poet_merge_step`` in the registry's sorted apply order, so for a
POET run the per-step merge has already folded ``R`` into the base weight when we
read it -> ``module.weight`` equals the effective weight ``W_eff``. For Adam/Muon
the weight IS the trained parameter. We register ``targets=()`` (like
``poet_grad_conditioning``) so we compose with ``poet_merge_step``'s wrapper of
the same symbol instead of raising ``PatchConflict``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# module-name suffix -> short matrix-type label. Covers both the fused mcore
# names and the unfused variants produced by --unfuse-qkv / --unfuse-fc1 (used by
# some POET configs, e.g. head-aligned attention). A module name ends in exactly
# one of these, so endswith matching is unambiguous.
_SUFFIX_TO_TYPE = {
    "self_attention.linear_qkv": "qkv",
    "self_attention.linear_q": "q",
    "self_attention.linear_k": "k",
    "self_attention.linear_v": "v",
    "self_attention.linear_proj": "proj",
    "mlp.linear_fc1": "fc1",
    "mlp.linear_fc1_gate": "fc1_gate",
    "mlp.linear_fc1_up": "fc1_up",
    "mlp.linear_fc2": "fc2",
}
_LAYER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.")

# warn-once state for the POET merge_period=0 (frozen-base) corner case
_state = {"warned_merge0": False}


def parse_layer_selection(spec, num_layers: int) -> set[int]:
    """Parse a layer spec into a set of layer indices.

    Tokens are comma-separated; each is one of ``first`` (0), ``mid``
    (num_layers // 2), ``last`` (num_layers - 1), or an integer index (negatives
    wrap from the end). Out-of-range integer indices are silently dropped.
    """
    out: set[int] = set()
    for raw in str(spec).split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "first":
            out.add(0)
        elif tok == "mid":
            out.add(num_layers // 2)
        elif tok == "last":
            out.add(num_layers - 1)
        else:
            idx = int(tok)
            if idx < 0:
                idx += num_layers
            if 0 <= idx < num_layers:
                out.add(idx)
    return out


def classify_linear(name: str):
    """Map a module name to ``(layer_index, matrix_type)`` or ``None``.

    Matches the transformer linears by suffix — fused (qkv/proj/fc1/fc2) and the
    unfused variants (q/k/v, fc1_gate/fc1_up). The POET base-weight child
    (``...linear_qkv.poet_linear``), embeddings, lm_head and norms do not match.
    """
    m = _LAYER_RE.search(name)
    if not m:
        return None
    for suffix, mtype in _SUFFIX_TO_TYPE.items():
        if name.endswith(suffix):
            return int(m.group(1)), mtype
    return None


def should_log(iteration: int, interval: int, *, poet: bool, merge_period: int) -> bool:
    """Decide whether to log on this step.

    Non-POET: every ``interval`` steps. POET: only on merge-boundary steps
    (``iteration % merge_period == 0``) that are also interval steps, because the
    raw base weight equals ``W_eff`` only right after a merge. POET with
    ``merge_period <= 0`` (frozen base) never logs.
    """
    if iteration <= 0 or interval <= 0:
        return False
    if poet:
        if merge_period <= 0:
            return False
        if iteration % merge_period != 0:
            return False
    return iteration % interval == 0
