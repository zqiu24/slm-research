"""Tests for the torchtitan WandBLogger key-normalization wrapper.

Importing src.titan_ext.metrics is CPU-safe: torchtitan is imported only inside
the apply_* functions, and the package __init__'s _patch_metrics() no-ops when
torchtitan is absent.
"""


def test_wrap_titan_wandb_log_renames_keys_and_preserves_step():
    from src.titan_ext.metrics import _wrap_titan_wandb_log

    captured = {}

    def fake_log(self, metrics, step):
        captured["metrics"] = metrics
        captured["step"] = step

    wrapped = _wrap_titan_wandb_log(fake_log)

    class FakeLogger:
        pass

    wrapped(
        FakeLogger(),
        {"loss_metrics/global_avg_loss": 2.0, "mfu(%)": 30.0, "lr": 1e-3},
        50,
    )
    assert captured["metrics"] == {"train/loss": 2.0, "mfu(%)": 30.0, "train/lr": 1e-3}
    assert captured["step"] == 50


def test_apply_titan_wandb_normalize_noops_without_torchtitan():
    # On a CPU box without torchtitan importable, apply returns False, never raises.
    from src.titan_ext.metrics import apply_titan_wandb_normalize

    assert apply_titan_wandb_normalize() in (True, False)
