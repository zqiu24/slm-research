"""Re-initialize a Megatron GPT model to torchtitan's native llama3 init scheme.

Why this exists
---------------
The torchtitan backend builds llama3 from its NATIVE model flavor, whose weight
initialization differs structurally from Megatron's. To reproduce a torchtitan
training curve on the Megatron backend we overwrite Megatron's post-build weights
with torchtitan's exact per-weight std recipe, transcribed from
``third_party/torchtitan/torchtitan/models/llama3/model/model.py``:

    token embeddings        : normal(0, 1.0)                         (nn.init.normal_)
    attn wq/wk/wv, ffn gate : trunc_normal(0, 0.02)                  (a=-2, b=2)
    attn wo, ffn up, ffn down: trunc_normal(0, 0.02 / sqrt(2*(layer+1)))
    final LM head           : trunc_normal(0, dim**-0.5, a=-3*std, b=3*std)
    RMSNorm weights         : left as built (Megatron inits to ones; torchtitan
                              resets norms to ones — equivalent, so untouched)

Note the SwiGLU gate/up asymmetry: torchtitan inits the *gate* (w1) at a fixed
0.02 but the *up* (w3) at the depth-scaled std — same as the *down* (w2). We
mirror that exactly (see model.py FeedForward.init_weights).

Applied AFTER the unfuse transform (so q/k/v and gate/up exist as separate
projections — matching torchtitan's separate wq/wk/wv and w1/w2/w3) and BEFORE
DDP / Float16Module / optimizer setup, by wrapping ``pretrain_gpt.model_provider``.

Determinism: requires TP=PP=1 (the regime ``unfuse_linears`` also requires). All
data-parallel replicas re-init identically because we seed a fixed value and walk
parameters in a deterministic (name-sorted) order; each weight's std is derived
from its own parsed layer index, not iteration order. We save and restore the
ambient RNG so Megatron's own RNG stream is left untouched.

CPU-importable (no Megatron import at module load); the model-provider wrapper
imports ``megatron.training.get_args`` lazily.
"""

from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Matches the per-layer index in Megatron param names, e.g.
# "decoder.layers.7.self_attention.linear_proj.weight" -> 7.
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    """Truncated-normal init, transcribed verbatim from torchtitan's helper
    (``third_party/torchtitan/torchtitan/models/utils.py``): draw in float32 to
    avoid bf16 instability, then copy back into the original dtype. ``a``/``b``
    are ABSOLUTE bounds (not multiples of std), exactly as torchtitan calls it.
    """
    tmp = tensor.float()
    nn.init.trunc_normal_(tmp, mean=mean, std=std, a=a, b=b)
    tensor.copy_(tmp)


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _init_param(name: str, weight: torch.Tensor, *, hidden_size: int) -> str | None:
    """Re-init one parameter tensor in-place per torchtitan's llama3 recipe.

    ``weight`` must be the raw data tensor (pass ``param.data``). Returns a short
    category tag for logging, or ``None`` if the parameter is intentionally left
    as Megatron built it (layernorms, biases, non-matching params).
    """
    # Token embeddings -> plain normal(0, 1.0) (NOT truncated, std=1.0).
    if name.endswith("word_embeddings.weight"):
        nn.init.normal_(weight)
        return "embed"
    # Final LM head (untied) -> trunc_normal(0, dim**-0.5), truncated at +/-3 std.
    if name.endswith("output_layer.weight"):
        std = hidden_size**-0.5
        trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        return "lm_head"

    layer = _layer_index(name)
    if layer is None:
        # decoder.final_layernorm / embedding norms / anything outside a block.
        return None
    depth_std = 0.02 / math.sqrt(2.0 * (layer + 1))

    # Fan-in projections initialised at a fixed 0.02.
    if (
        name.endswith("linear_q.weight")
        or name.endswith("linear_k.weight")
        or name.endswith("linear_v.weight")
        or name.endswith("linear_qkv.weight")  # fused fallback: whole matrix at 0.02
        or name.endswith("linear_fc1_gate.weight")
    ):
        trunc_normal_(weight, mean=0.0, std=0.02)
        return "fanin_0.02"

    # Depth-scaled output / up / down projections.
    if (
        name.endswith("linear_proj.weight")  # attention output (wo)
        or name.endswith("linear_fc1_up.weight")  # SwiGLU up (w3) — depth-scaled like w2
        or name.endswith("linear_fc2.weight")  # SwiGLU down (w2)
    ):
        trunc_normal_(weight, mean=0.0, std=depth_std)
        return "depth_scaled"

    # Fused-fc1 fallback (titan_init WITHOUT --unfuse-fc1): split [gate; up].
    if name.endswith("linear_fc1.weight"):
        out_f = weight.shape[0]
        if out_f % 2 != 0:
            raise ValueError(f"titan_init: fused linear_fc1 out dim {out_f} is not even")
        ffn = out_f // 2
        trunc_normal_(weight[:ffn], mean=0.0, std=0.02)  # gate (w1) fixed 0.02
        trunc_normal_(weight[ffn:], mean=0.0, std=depth_std)  # up (w3) depth-scaled
        return "fused_fc1"

    # input_layernorm / pre_mlp_layernorm / etc. — leave as built (ones).
    return None


def apply_titan_init(model: nn.Module, *, hidden_size: int, num_layers: int, seed: int) -> dict:
    """Overwrite ``model``'s weights with torchtitan's llama3 init, in-place.

    Seeds a fixed value so every data-parallel replica produces identical weights,
    then restores the ambient RNG so Megatron's stream is unperturbed. ``num_layers``
    is accepted for symmetry/logging; the per-layer std is derived from each param's
    own name. Returns a count of re-initialised params per category.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    counts: dict[str, int] = {}
    try:
        torch.manual_seed(int(seed))
        # Name-sorted so the RNG-draw order is identical on every rank.
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            tag = _init_param(name, p.data, hidden_size=hidden_size)
            if tag is not None:
                counts[tag] = counts.get(tag, 0) + 1
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    logger.info(
        "[titan_init] re-initialized weights to torchtitan llama3 scheme "
        "(hidden=%d, layers=%d, seed=%d): %s",
        hidden_size,
        num_layers,
        int(seed),
        counts,
    )
    return counts


def wrap_model_provider(orig_provider):
    """Wrap a Megatron ``model_provider`` so the built model is re-initialized to
    torchtitan's scheme before DDP/optimizer setup. Reads dims/seed from the parsed
    Megatron args and enforces TP=PP=1.
    """

    def _wrapped(*args_, **kwargs_):
        from megatron.training import get_args

        model = orig_provider(*args_, **kwargs_)
        args = get_args()
        tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
        pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
        if tp != 1 or pp != 1:
            raise ValueError(
                f"[titan_init] requires TP=PP=1 (got tp={tp}, pp={pp}); "
                "per-shard / per-stage re-init is not implemented."
            )
        apply_titan_init(
            model,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            seed=int(getattr(args, "seed", 1234) or 1234),
        )
        return model

    return _wrapped
