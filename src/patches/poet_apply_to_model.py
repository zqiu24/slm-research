"""Patch: replace ParallelLinear modules with POETMegatronLinear at model build.

Targets ``megatron.training.training.get_model``, but applies POET by wrapping
the ``model_provider_func`` *argument* rather than transforming get_model's
return value. This is deliberate:

    get_model(model_provider_func, ...) builds each chunk via
    model_provider_func(...), then .cuda()s it, wraps it in Float16Module, and
    finally wraps it in DistributedDataParallel (DDP). DDP's __init__ snapshots
    the model's ``requires_grad=True`` params into its contiguous grad buffer
    (.main_grad slices) -- ONCE. Any param added *after* that snapshot has no
    .main_grad and its gradient falls back to plain autograd .grad, which means
    it MISSES the ``1/num_tokens`` normalization that finalize_model_grads
    applies via DDP.scale_gradients() (buffers only). It also misses the DP
    all-reduce and gradient clipping that flow through the buffer.

    The earlier design applied POET to get_model's *return value* (post-DDP),
    so the freshly-created ``oft_R`` generators were never in the buffer. That
    made their raw grad norm ~num_tokens too large (e.g. ~62k-73k vs the correct
    ~0.5) while embeddings/norms (in the buffer) were normalized fine.

    By wrapping ``model_provider_func`` we run the POET replacement on the raw,
    pre-DDP chunk, so ``oft_R`` exists with requires_grad=True at DDP.__init__
    and becomes a first-class buffer citizen -- normalized, all-reduced, and
    clipped exactly like every other parameter.

Unfusing fused linears (qkv / fc1) is handled by the ``model_unfuse_linears``
patch, which wraps ``pretrain_gpt.model_provider`` directly. Because patches are
applied in sorted() order, ``model_unfuse_linears`` installs its model_provider
wrapper before this patch installs its get_model wrapper; at call time POET's
provider wrapper calls the (already unfuse-wrapped) provider, so POET runs AFTER
unfuse and naturally wraps the unfused linears. We keep targeting ``get_model``
(not ``pretrain_gpt.model_provider``) to avoid a target clash with
``model_unfuse_linears`` in the patch registry.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.get_model",)
logger = logging.getLogger(__name__)

# Manual switch for the per-parameter dump emitted after POET wraps the model
# (name / shape / requires_grad / numel / derived block_size). Flip to False to
# silence it on real runs; the one-line trainable/frozen summary always logs.
_DUMP_POET_PARAMS = True


@register_patch(name="poet_apply_to_model", targets=_TARGET)
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    from src.optim.poet_layers import replace_linears_with_poet

    _orig = _mt.get_model

    def _apply_poet_to_chunk(m, args) -> int:
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        return replace_linears_with_poet(
            m,
            block_size=block,
            block_count=block_count,
            init_type=init,
            mup_alpha=mup_alpha,
            cache_mode=cache_mode,
        )

    def _wrapped(*a, **kw):
        args = get_args()
        if not getattr(args, "poet", False):
            return _orig(*a, **kw)

        import os

        # Degenerate control (POET_WRAP_NONE=1): wrap ZERO layers but keep the
        # whole POET path otherwise identical (unfuse, optimizer setup, patch
        # set). The unfused linears stay trainable and flow through POETAdam, so
        # with optim.poet.scale=1.0 + merges disabled this run == plain Adam.
        wrap_none = os.environ.get("POET_WRAP_NONE") == "1"
        if wrap_none:
            logger.warning(
                "[POET] POET_WRAP_NONE=1 -> wrapping ZERO layers "
                "(degenerate Adam control); linears remain trainable."
            )

        counter = {"total": 0}

        def _make_poet_provider(mpf):
            # Wrap the model_provider_func so POET is applied to each freshly
            # built chunk BEFORE get_model casts it (.cuda/Float16Module) and
            # wraps it in DDP -- so oft_R is present at DDP.__init__ and lands
            # in the grad buffer.
            def _poet_provider(*pa, **pkw):
                m = mpf(*pa, **pkw)
                if not wrap_none:
                    counter["total"] += _apply_poet_to_chunk(m, args)
                return m

            return _poet_provider

        # model_provider_func is the first positional arg to get_model (Megatron
        # always calls it positionally); fall back to the keyword form.
        if a:
            new_args = (_make_poet_provider(a[0]), *a[1:])
            model = _orig(*new_args, **kw)
        elif "model_provider_func" in kw:
            kw = dict(kw)
            kw["model_provider_func"] = _make_poet_provider(kw["model_provider_func"])
            model = _orig(*a, **kw)
        else:
            logger.warning(
                "[POET] could not locate model_provider_func in get_model args; "
                "POET was NOT applied."
            )
            return _orig(*a, **kw)

        # ---- logging (on the final, DDP-wrapped model) ----
        chunks = model if isinstance(model, list) else [model]
        import torch

        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_dist else 0

        if _DUMP_POET_PARAMS and rank == 0 and not wrap_none:
            import math

            def _block_size_from_oft_r(p):
                # oft_R_in/oft_R_out have shape (n_blocks, n_elems) where
                # n_elems = b*(b-1)/2 is the count of strictly-upper-triangular
                # entries of a b x b skew block. Invert for b:
                #   8*n_elems + 1 = (2b - 1)^2  =>  b = (1 + sqrt(8*n_elems + 1)) / 2
                n_elems = p.shape[-1]
                b = (1 + math.isqrt(8 * int(n_elems) + 1)) // 2
                return b, int(p.shape[0])  # (block_size, n_blocks)

            print(
                "[POET] ===== parameter dump "
                "(name | shape | requires_grad | numel | block_size x n_blocks) =====",
                flush=True,
            )
            for m in chunks:
                for pname, p in m.named_parameters():
                    extra = ""
                    if "oft_R" in pname and p.dim() == 2:
                        block_size, n_blocks = _block_size_from_oft_r(p)
                        extra = f" block_size={block_size} n_blocks={n_blocks}"
                    print(
                        f"[POET] {pname:<78} {tuple(p.shape)!s:<22} "
                        f"requires_grad={p.requires_grad} numel={p.numel()}{extra}",
                        flush=True,
                    )
            print("[POET] ===== end parameter dump =====", flush=True)

        trainable = sum(p.numel() for m in chunks for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for m in chunks for p in m.parameters() if not p.requires_grad)
        ratio = trainable / max(trainable + frozen, 1) * 100
        if _DUMP_POET_PARAMS and rank == 0:
            print(
                f"[POET] replaced {counter['total']} linears | trainable={trainable} "
                f"frozen={frozen} ({ratio:.2f}%) [applied pre-DDP -> oft_R in grad buffer]",
                flush=True,
            )
        logger.info(
            "[POET] replaced %d linears | trainable=%d frozen=%d (%.2f%%) [pre-DDP]",
            counter["total"],
            trainable,
            frozen,
            ratio,
        )
        return model

    _mt.get_model = _wrapped
