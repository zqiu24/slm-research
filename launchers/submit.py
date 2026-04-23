"""Launcher entry point: resolves the six-axis config, runs guardrails,
archives the resolved config, and submits to SLURM.

See SPEC.md §5.2 and §9 for the required pipeline. SLURM submission itself
is stubbed until the training loop lands; until then, ``--dry-run`` prints
the resolved config and exits, which is what CI uses.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.precision.capability import assert_compatible
from src.utils.config_diff import diff_from_champion
from src.utils.config_hash import config_hash
from src.utils.git_meta import git_sha, submodule_sha
from src.utils.ladder_math import total_tokens
from src.utils.parallelism import resolve as resolve_parallelism

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_champion_for(family: str, scale: str) -> DictConfig:
    """Load the champion config layered onto the same family + scale."""
    base_family = OmegaConf.load(REPO_ROOT / f"configs/base/family/{family}.yaml")
    base_scale = OmegaConf.load(REPO_ROOT / f"configs/base/scale/{scale}.yaml")
    champion = OmegaConf.load(REPO_ROOT / "configs/experiments/champion.yaml")
    return OmegaConf.merge(base_family, base_scale, champion)


def _first_line(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().splitlines()[0]


def _write_if_new(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text()
        if existing != content:
            raise RuntimeError(
                f"Refusing to overwrite existing archived config at {path}. "
                f"Two different resolved configs hashed to the same value -> bug."
            )
        return
    path.write_text(content)


def _append_launch_metadata(path: Path, cfg: DictConfig, *, seed: int, job_id: str | None) -> None:
    entry = {
        "seed": seed,
        "git_sha": cfg._derived.get("git_sha"),
        "megatron_sha": cfg._derived.get("megatron_sha"),
        "patch_set_hash": cfg._derived.get("patch_set_hash"),
        "dataset_hash": cfg._derived.get("dataset_hash"),
        "launch_timestamp_utc": cfg._derived.get("launch_timestamp_utc"),
        "cluster": cfg.cluster.name,
        "wandb_project": cfg.wandb.project,
        "job_id": job_id,
    }
    entries = []
    if path.exists():
        entries = json.loads(path.read_text())
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=2))


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Run the full resolution pipeline; mutate and return ``cfg``."""
    # 1. Parallelism
    world_nodes = cfg.cluster.nodes or 1
    pspec = resolve_parallelism(
        int(cfg.base.non_embedding_params),
        cluster_nodes=int(world_nodes),
        gpus_per_node=int(cfg.cluster.gpus_per_node),
        tp_size_rules=list(cfg.parallelism.tp_size_rules),
    )
    cfg.parallelism.tp = pspec.tp
    cfg.parallelism.pp = pspec.pp
    cfg.parallelism.dp = pspec.dp

    # 2. Embedding math — deferred until the dataset manifest is available.
    # For now we stamp a placeholder vocab_size so ladder math still runs.
    # The actual manifest check happens in the real launcher (step 7 below).
    vocab = cfg.base.tokenizer.get("nominal_vocab_size", 0)
    hidden = int(cfg.base.model.hidden_size)
    cfg._derived.embedding_params = vocab * hidden
    cfg._derived.lm_head_params = 0 if cfg.base.model.tie_embeddings else vocab * hidden
    cfg._derived.total_params = (
        int(cfg.base.non_embedding_params)
        + int(cfg._derived.embedding_params)
        + int(cfg._derived.lm_head_params)
    )

    # 3. Token count
    cfg.training.total_tokens = total_tokens(
        cfg.training.tokens_per_param, int(cfg.base.non_embedding_params)
    )

    # 4. Capability check
    assert_compatible(
        required=list(cfg.experiment.get("required_capabilities", []) or []),
        available=list(cfg.cluster.capabilities),
        cluster_name=str(cfg.cluster.name),
    )

    # 5. Version pins — stubbed. TODO: cross-check installed packages.
    # assert_version_matches("transformer_engine", cfg.cluster.transformer_engine_version)
    # assert_version_matches("torch", cfg.cluster.pytorch_version)

    # 6. Dataset manifest — stubbed. TODO: verify SHA against cfg.data.path manifest.
    cfg._derived.dataset_hash = cfg.data.get("expected_manifest_hash") or "unverified"

    # 7. Patches — stubbed. TODO: import patches referenced in cfg.experiment.patches
    # so @register_patch decorators run, then call apply_patches.
    cfg._derived.patch_set_hash = "noop" + "0" * 12

    # 8. Git metadata
    allow_dirty = bool(cfg.get("allow_dirty", False)) or str(cfg.wandb.project).startswith(
        "sandbox"
    )
    cfg._derived.git_sha = git_sha(cwd=REPO_ROOT, allow_dirty=allow_dirty)
    try:
        cfg._derived.megatron_sha = submodule_sha("third_party/Megatron-LM", cwd=REPO_ROOT)
    except Exception:
        cfg._derived.megatron_sha = "unpinned"

    # 9. Config hash (seed excluded)
    cfg._derived.config_hash = config_hash(cfg)

    # 10. Config diff from champion
    champion_cfg = _load_champion_for(str(cfg.base.family), str(cfg.base.scale))
    cfg._derived.config_diff_from_champion = diff_from_champion(cfg, champion_cfg)
    cfg._derived.experiment_summary = _first_line(cfg.experiment.description)

    # 11. Launch metadata
    cfg._derived.launch_timestamp_utc = datetime.now(timezone.utc).isoformat()
    return cfg


def archive_resolved_config(cfg: DictConfig) -> Path:
    """Write ``runs/<config_hash>/resolved_config.yaml`` append-only."""
    archive_dir = REPO_ROOT / "runs" / str(cfg._derived.config_hash)
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_if_new(archive_dir / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
    _append_launch_metadata(
        archive_dir / "launch_metadata.json",
        cfg,
        seed=int(cfg.seed),
        job_id=None,
    )
    return archive_dir


def submit(cfg: DictConfig, *, dry_run: bool) -> int | None:
    """Resolve, archive, and (eventually) submit to SLURM. Returns job id or None."""
    resolve_config(cfg)
    archive_resolved_config(cfg)
    if dry_run:
        return None
    raise NotImplementedError(
        "SLURM submission is not wired up yet; re-run with --dry-run or see SPEC.md §12 step 10."
    )


def _parse_overrides(pairs: list[str]) -> DictConfig:
    base = OmegaConf.load(REPO_ROOT / "configs/launch/config.yaml")
    # For each override like ``base/family=qwen3``, interpret keys with '/'
    # as defaults-list selections and load the corresponding file; plain
    # ``k=v`` pairs become dotlist overrides.
    dotlist: list[str] = []
    merges: list[DictConfig] = []
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Bad override {raw!r}; expected KEY=VALUE")
        key, value = raw.split("=", 1)
        if "/" in key:
            group = key.strip()
            candidate = REPO_ROOT / "configs" / group / f"{value}.yaml"
            if not candidate.exists():
                raise FileNotFoundError(f"No such config: {candidate}")
            merges.append(OmegaConf.load(candidate))
        else:
            dotlist.append(raw)
    resolved = OmegaConf.merge(base, *merges)
    if dotlist:
        resolved = OmegaConf.merge(resolved, OmegaConf.from_dotlist(dotlist))
    return resolved  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="axis=name or dotted.key=value pairs")
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve + archive; skip SLURM submission"
    )
    args = parser.parse_args()

    cfg = _parse_overrides(args.overrides)
    # allow_dirty is on in sandbox projects by default
    if "allow_dirty" not in cfg:
        cfg.allow_dirty = False

    job_id = submit(cfg, dry_run=args.dry_run)
    archive = REPO_ROOT / "runs" / str(cfg._derived.config_hash)
    print(
        json.dumps(
            {
                "config_hash": str(cfg._derived.config_hash),
                "config_diff_from_champion": str(cfg._derived.config_diff_from_champion),
                "total_tokens": int(cfg.training.total_tokens),
                "parallelism": {
                    "tp": int(cfg.parallelism.tp),
                    "pp": int(cfg.parallelism.pp),
                    "dp": int(cfg.parallelism.dp),
                },
                "archive": os.fspath(archive),
                "job_id": job_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
