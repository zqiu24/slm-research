"""Tests for the canonical W&B metric schema + normalize()."""

from src.utils.wandb_metrics import (
    CORE_CANONICAL,
    MEGATRON_TO_CANONICAL,
    TITAN_TO_CANONICAL,
    normalize,
)


def test_megatron_core_keys_map_to_canonical():
    out = normalize({"lm loss": 2.5, "learning-rate": 1e-3, "grad-norm": 1.2}, "megatron")
    assert out == {"train/loss": 2.5, "train/lr": 1e-3, "train/grad_norm": 1.2}


def test_megatron_validation_loss_maps_to_val_loss():
    assert normalize({"lm loss validation": 3.0}, "megatron") == {"val/loss": 3.0}


def test_megatron_unmapped_keys_pass_through():
    out = normalize({"params-norm": 9.0, "num-zeros": 4, "grad-norm-clipped": 1.0}, "megatron")
    assert out == {"params-norm": 9.0, "num-zeros": 4, "grad-norm-clipped": 1.0}


def test_megatron_throughput_is_not_remapped_to_tps():
    # Megatron 'throughput' is TFLOP/s/GPU, not tokens/sec — must stay native.
    assert normalize({"throughput": 177.2}, "megatron") == {"throughput": 177.2}


def test_torchtitan_core_keys_map_to_canonical():
    out = normalize(
        {
            "loss_metrics/global_avg_loss": 2.0,
            "loss_metrics/global_max_loss": 2.4,
            "lr": 1e-3,
            "grad_norm": 1.1,
            "n_tokens_seen": 4096,
            "time_metrics/end_to_end(s)": 0.5,
        },
        "torchtitan",
    )
    assert out == {
        "train/loss": 2.0,
        "train/loss_max": 2.4,
        "train/lr": 1e-3,
        "train/grad_norm": 1.1,
        "train/tokens_seen": 4096,
        "perf/step_time_s": 0.5,
    }


def test_torchtitan_validation_loss_maps_to_val_loss():
    assert normalize({"validation_metrics/loss": 3.1}, "torchtitan") == {"val/loss": 3.1}


def test_torchtitan_throughput_tps_is_not_normalized():
    # torchtitan tps is normalized by non_data_parallel_size; a Megatron-computed
    # tokens/sec is the global aggregate. Different quantities -> stay native.
    assert normalize({"throughput(tps)": 8000.0}, "torchtitan") == {"throughput(tps)": 8000.0}


def test_torchtitan_unmapped_keys_pass_through():
    out = normalize({"mfu(%)": 30.0, "tflops": 120.0, "memory/max_active(GiB)": 40.0}, "torchtitan")
    assert out == {"mfu(%)": 30.0, "tflops": 120.0, "memory/max_active(GiB)": 40.0}


def test_normalize_is_idempotent_on_canonical_input():
    canonical = {"train/loss": 2.0, "perf/step_time_s": 0.5}
    assert normalize(canonical, "megatron") == canonical
    assert normalize(canonical, "torchtitan") == canonical


def test_normalize_empty_and_none():
    assert normalize({}, "megatron") == {}
    assert normalize(None, "torchtitan") == {}


def test_both_backends_cover_the_core_set():
    # Every CORE_CANONICAL key (except the two computed-only on megatron) must be a
    # target of at least one backend map.
    targets = set(MEGATRON_TO_CANONICAL.values()) | set(TITAN_TO_CANONICAL.values())
    targets |= {"val/loss"}  # produced by the validation-suffix rule
    assert targets >= CORE_CANONICAL
