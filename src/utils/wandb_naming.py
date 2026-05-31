"""Unified W&B run-name builder shared by both training backends.

Both the Megatron and the torchtitan launcher derive the SAME canonical run name
(experiment + arch + optimizer/LR) and tag it with a ``[<backend>]`` prefix, so a
single W&B dashboard can tell the two backends apart while everything else about
the name stays identical. This is the single source of truth — the launchers call
``wandb_run_name`` rather than each rolling their own naming.
"""

from __future__ import annotations

from omegaconf import DictConfig


def wandb_base_name(cfg: DictConfig) -> str:
    """Canonical, backend-independent run name: ``<experiment>-<family>-<scale>-lr<lr>``.

    No seed, no timestamp (run-to-run identity lives in the on-disk run dir and
    W&B's own run id). ``lr`` is ``optim.lr`` for adam/poet/ngpt. For muon_hybrid
    it is the Adam-side LR (``optim.adam.lr``) and a second ``-muon_lr<lr>`` segment
    holds the Muon-side LR (``optim.muon.lr``). POET emits the block
    parameterization BEFORE the LR segment (``-bc<n>`` when ``block_count`` is set,
    else ``-bs<n>`` for the legacy shared block size) and trails it with the oft_R
    LR multiplier ``-scale<v>`` (``optim.poet.scale``), i.e.
    ``...-<bc|bs>-lr<lr>-scale<v>``, so block/scale sweeps are distinguishable.
    """
    optim = cfg.optim
    otype = str(optim.type)

    parts = [
        str(cfg.experiment.name),
        str(cfg.base.family),
        str(cfg.base.scale),
    ]
    # POET: block parameterization comes BEFORE the LR segment, giving
    # ...-<bc|bs>-lr<lr>-scale<v>.
    poet = optim.get("poet", {}) if otype == "poet" else {}
    if otype == "poet":
        block_count = poet.get("block_count", None)
        if block_count is not None:
            parts.append(f"bc{int(block_count)}")
        else:
            parts.append(f"bs{int(poet.get('block_size', 256))}")

    if otype == "muon_hybrid":
        adam_lr = optim.get("adam", {}).get("lr", 1.0e-3)
        muon_lr = optim.get("muon", {}).get("lr", adam_lr)
        parts.append(f"lr{float(adam_lr):g}")
        parts.append(f"muon_lr{float(muon_lr):g}")
    else:
        lr = optim.get("lr", optim.get("adam", {}).get("lr", 1.0e-3))
        parts.append(f"lr{float(lr):g}")

    if otype == "poet":
        # oft_R LR multiplier (optim.poet.scale) trails the LR segment.
        parts.append(f"scale{float(poet.get('scale', 1.0)):g}")
    return "-".join(parts)


def wandb_run_name(cfg: DictConfig) -> str:
    """Canonical name tagged with the backend: ``[<backend>] <base>``.

    The ONLY difference between the two backends' W&B names is this prefix; the rest
    is byte-identical (same `wandb_base_name`). ``backend`` defaults to ``megatron``
    when unset, matching `launchers.submit.resolve_config`.
    """
    backend = str(cfg.get("backend", "megatron"))
    return f"[{backend}] {wandb_base_name(cfg)}"
