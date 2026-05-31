"""titan_ext metrics patch helpers: ETA formatting + rank-0 logger proxy.

CPU-only: covers the pure pieces of src/titan_ext/metrics.py. The actual
MetricsProcessor.log monkeypatch needs torchtitan + a live process group, so it
is exercised by a real torchtitan run, not here.
"""

from __future__ import annotations

from src.titan_ext.metrics import _RankZeroEtaLogger, format_eta


class _FakeLogger:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)


def test_format_eta_matches_megatron_style():
    assert format_eta(100, 1.5) == "0h02m"  # 150s
    assert format_eta(0, 5.0) == "0h00m"
    assert format_eta(3600, 2.0) == "2h00m"  # 7200s
    assert format_eta(100, 95.0) == "2h38m"  # 9500s
    # Negative / past-the-end clamps to zero.
    assert format_eta(-5, 1.0) == "0h00m"


def test_proxy_appends_eta_on_rank0():
    real = _FakeLogger()
    proxy = _RankZeroEtaLogger(real, eta_str="1h30m", is_rank0=True)
    proxy.info("step: 10  loss: 1.23  mfu: 1.67%")
    assert real.infos == ["step: 10  loss: 1.23  mfu: 1.67%  ETA: 1h30m"]


def test_proxy_suppresses_off_rank0():
    real = _FakeLogger()
    proxy = _RankZeroEtaLogger(real, eta_str="1h30m", is_rank0=False)
    proxy.info("step: 10  loss: 1.23")
    assert real.infos == []


def test_proxy_no_eta_when_empty():
    real = _FakeLogger()
    proxy = _RankZeroEtaLogger(real, eta_str="", is_rank0=True)
    proxy.info("step: 10  loss: 1.23")
    assert real.infos == ["step: 10  loss: 1.23"]


def test_proxy_forwards_other_methods():
    real = _FakeLogger()
    proxy = _RankZeroEtaLogger(real, eta_str="1h30m", is_rank0=True)
    proxy.warning("heads up")
    assert real.warnings == ["heads up"]
