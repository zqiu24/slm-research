"""Data cache must be keyed on dataset identity, not run identity."""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.megatron_args import build_megatron_args


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_data_cache_path_is_dataset_keyed_not_config_keyed():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=1_2b",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
            "data=nemotron_cc_v2_llama31_8b",
        ]
    )
    resolve_config(cfg)

    args = build_megatron_args(cfg)
    cache = _arg_value(args, "--data-cache-path")

    assert str(cfg._derived.run_name) not in cache, "data cache must NOT be keyed on run identity"
    assert cfg.data.name in cache, f"data cache must reference dataset name; got {cache!r}"


def test_data_cache_is_stable_across_optim_changes():
    common = [
        "base/family=llama3",
        "base/scale=1_2b",
        "training_regime=ablation_20x",
        "cluster=h800_cn",
        "data=nemotron_cc_v2_llama31_8b",
    ]
    a = _parse_overrides([*common, "experiment=champion"])
    b = _parse_overrides([*common, "experiment=optim/muon_hybrid"])
    resolve_config(a)
    resolve_config(b)

    cache_a = _arg_value(build_megatron_args(a), "--data-cache-path")
    cache_b = _arg_value(build_megatron_args(b), "--data-cache-path")
    assert cache_a == cache_b, "switching optimiser must NOT invalidate data cache"
