"""The CROSS-SIDE DECORRELATION startup banner (the sweep scripts' tripwire)."""

from __future__ import annotations

import logging
import types

from src.optim.poet import _log_decorrelate_banner


def test_banner_reports_lambda_and_mode(caplog):
    cfg = types.SimpleNamespace(
        poet_lie_alternating=True,
        poet_lie_ortho_decorrelate_mode="symmetric",
        poet_lie_ortho_decorrelate_lambda=0.5,
        poet_lie_ortho_decorrelate_renorm=True,
        poet_lie_ortho_decorrelate_cos_threshold=0.0,
    )
    logger = logging.getLogger("test.decorr.banner")
    with caplog.at_level(logging.WARNING, logger="test.decorr.banner"):
        _log_decorrelate_banner(cfg, logger)
    msg = caplog.text
    assert "CROSS-SIDE DECORRELATION ON" in msg
    assert "mode=symmetric" in msg
    assert "lambda=0.5" in msg
    assert "renorm=True" in msg
    assert "alternating=True" in msg
