"""Unit tests for src.utils.config_hash (SPEC.md §5.3, §10.1).

Invariants:
- Same resolved config -> same hash (determinism).
- Different seeds -> same hash (seed is excluded).
- Different cluster-realization fields -> same hash (parallelism is excluded).
- Different experiment -> different hash.
- Reordering dicts must not change the hash (sort-key stability).
"""

from __future__ import annotations

from src.utils.config_hash import config_hash


def _base_cfg() -> dict:
    return {
        "base": {"family": "qwen3", "scale": "1_2b", "non_embedding_params": 1_200_000_000},
        "experiment": {"name": "baseline", "family": "optim"},
        "optim": {"type": "adamw", "lr": 3e-4},
        "training": {"tokens_per_param": 20, "global_batch_size_tokens": 4_194_304},
        "cluster": {
            "name": "h800_cn",
            "nodes": 6,
            "slurm_partition": "h800",
            "slurm_account": "research",
            "capabilities": ["bf16", "fp8"],
        },
        "parallelism": {"tp": 1, "pp": 1, "dp": 48},
        "seed": 42,
        "wandb": {"project": "pretrain-ablations-1_2b"},
        "_derived": {"git_sha": "abc123"},
    }


def test_hash_is_deterministic():
    assert config_hash(_base_cfg()) == config_hash(_base_cfg())


def test_hash_is_stable_across_key_order():
    cfg_a = _base_cfg()
    cfg_b = {k: cfg_a[k] for k in reversed(list(cfg_a))}
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_seed_does_not_affect_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["seed"] = 7
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_cluster_nodes_does_not_affect_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["cluster"]["nodes"] = 2
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_parallelism_does_not_affect_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["parallelism"] = {"tp": 4, "pp": 2, "dp": 6}
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_wandb_does_not_affect_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["wandb"]["project"] = "sandbox-alice"
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_experiment_affects_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["optim"]["type"] = "muon_hybrid"
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_family_affects_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["base"]["family"] = "llama3"
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_scale_affects_hash():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["base"]["scale"] = "2_4b"
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_hash_length():
    assert len(config_hash(_base_cfg())) == 16
