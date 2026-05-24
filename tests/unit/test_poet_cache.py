"""Unit tests for the POET Cayley-Neumann cache.

CPU-runnable tests cover the cache state machine, registry liveness,
dispatch routing, invalidation hooks, optimizer-hook installation, and
argv plumbing. GPU-required tests (numerical parity, DDP smokes) are
guarded by skipif and run on the cluster.
"""

import gc

import pytest

from src.optim import poet_cache as pc


def test_default_cache_mode_is_none():
    pc.reset_for_testing()
    assert pc.get_cache_mode() == "none"


def test_set_cache_mode_valid():
    pc.set_cache_mode("cached_fwd")
    assert pc.get_cache_mode() == "cached_fwd"
    pc.set_cache_mode("cached_fwd_bwd")
    assert pc.get_cache_mode() == "cached_fwd_bwd"
    pc.set_cache_mode("none")
    assert pc.get_cache_mode() == "none"


def test_set_cache_mode_rejects_unknown():
    with pytest.raises(ValueError, match="poet_cache_mode"):
        pc.set_cache_mode("bogus")


def test_version_starts_at_zero_and_bumps_monotonically():
    pc.reset_for_testing()
    assert pc.get_poet_version() == 0
    pc.bump_poet_version()
    assert pc.get_poet_version() == 1
    pc.bump_poet_version()
    assert pc.get_poet_version() == 2


def test_registry_holds_weakrefs():
    pc.reset_for_testing()

    class Dummy:
        pass

    d = Dummy()
    pc.register_poet_layer(d)
    assert list(pc.iter_live_layers()) == [d]
    del d
    gc.collect()
    assert list(pc.iter_live_layers()) == []


def test_iter_live_layers_skips_dead_refs():
    pc.reset_for_testing()

    class Dummy:
        pass

    alive = Dummy()
    dead = Dummy()
    pc.register_poet_layer(alive)
    pc.register_poet_layer(dead)
    del dead
    gc.collect()
    assert list(pc.iter_live_layers()) == [alive]
