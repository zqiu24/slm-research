"""Launcher entry point: resolves the six-axis config, runs guardrails,
archives the resolved config, and submits to SLURM.

See SPEC.md §5.2 and §9 for the required pipeline. SLURM submission itself
is stubbed until the training loop lands; until then, ``--dry-run`` prints
the resolved config and exits, which is what CI uses.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.patches import patch_set_hash
from src.precision.capability import assert_compatible
from src.utils.config_diff import diff_from_champion
from src.utils.git_meta import git_sha, submodule_sha
from src.utils.ladder_math import total_tokens
from src.utils.parallelism import resolve as resolve_parallelism

REPO_ROOT = Path(__file__).resolve().parent.parent


AXIS_TO_CONFIG_DIR = {
    "base/family": "configs/base/family",
    "base/scale": "configs/base/scale",
    "experiment": "configs/experiments",
    "training_regime": "configs/training_regime",
    "cluster": "configs/clusters",
    "data": "configs/data",
}


def _axis_config_path(axis: str, value: str) -> Path:
    if axis not in AXIS_TO_CONFIG_DIR:
        raise ValueError(f"Unknown config axis {axis!r}")
    path = REPO_ROOT / AXIS_TO_CONFIG_DIR[axis] / f"{value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No such config: {path}")
    return path


def _default_axis_entries(defaults: list) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for item in defaults:
        if item == "_self_":
            continue
        if isinstance(item, DictConfig):
            item = OmegaConf.to_container(item, resolve=True)
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Unsupported defaults entry: {item!r}")
        axis, value = next(iter(item.items()))
        entries.append((str(axis), str(value)))
    return entries


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


def _append_launch_metadata(path: Path, cfg: DictConfig, *, seed: int, job_id: str | None) -> None:
    entry = {
        "run_name": cfg._derived.get("run_name"),
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


def _register_experiment_patches(cfg: DictConfig) -> str:
    """Import patch modules named by cfg and return their deterministic hash.

    Training ranks apply the patches inside launchers.pretrain_gpt_slm. The
    parent launcher only records the hash, so dry-runs do not mutate Megatron.
    """
    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    for name in patches:
        importlib.import_module(f"src.patches.{name}")
    return patch_set_hash(patches)


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

    # 7. Patches — import each name listed in cfg.experiment.patches so the
    # @register_patch decorators run, then capture the deterministic hash.
    # The training ranks apply the patches inside pretrain_gpt_slm.
    cfg._derived.patch_set_hash = _register_experiment_patches(cfg)

    # 8. Git metadata
    allow_dirty = bool(cfg.get("allow_dirty", False)) or str(cfg.wandb.project).startswith(
        "sandbox"
    )
    cfg._derived.git_sha = git_sha(cwd=REPO_ROOT, allow_dirty=allow_dirty)
    try:
        cfg._derived.megatron_sha = submodule_sha("third_party/Megatron-LM", cwd=REPO_ROOT)
    except Exception:
        cfg._derived.megatron_sha = "unpinned"

    # 9. Config diff from champion
    champion_cfg = _load_champion_for(str(cfg.base.family), str(cfg.base.scale))
    cfg._derived.config_diff_from_champion = diff_from_champion(cfg, champion_cfg)
    cfg._derived.experiment_summary = _first_line(cfg.experiment.description)

    # 10. Run identity — a readable, timestamped name (one fresh dir per launch).
    now = datetime.now(timezone.utc)
    cfg._derived.launch_timestamp_utc = now.isoformat()
    run_name = (
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}"
        f"-s{cfg.seed}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    )
    cfg._derived.run_name = run_name
    cfg._derived.run_dir = f"runs/{run_name}"
    return cfg


def archive_resolved_config(cfg: DictConfig) -> Path:
    """Write ``runs/<run_name>/resolved_config.yaml`` and launch metadata.

    Each launch gets its own timestamped directory, so this is a plain write.
    """
    archive_dir = REPO_ROOT / cfg._derived.run_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
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
    defaults = list(base.pop("defaults", []) or [])

    axis_values = dict(_default_axis_entries(defaults))
    dotlist: list[str] = []

    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Bad override {raw!r}; expected KEY=VALUE")
        key, value = raw.split("=", 1)
        if key in AXIS_TO_CONFIG_DIR or key in {"base/family", "base/scale"}:
            axis_values[key] = value
        else:
            dotlist.append(raw)

    merges = [OmegaConf.load(_axis_config_path(axis, value)) for axis, value in axis_values.items()]
    resolved = OmegaConf.merge(*merges, base)
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
    archive = REPO_ROOT / cfg._derived.run_dir
    print(
        json.dumps(
            {
                "run_name": str(cfg._derived.run_name),
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
