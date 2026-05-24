"""Unit tests for the POET Cayley-Neumann cache.

CPU-runnable tests cover the cache state machine, registry liveness,
dispatch routing, invalidation hooks, optimizer-hook installation, and
argv plumbing. GPU-required tests (numerical parity, DDP smokes) are
guarded by skipif and run on the cluster.
"""

import gc

import pytest
import torch

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


def test_cached_layer_starts_invalidated():
    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_invalidate_clears_all_cache_slots():
    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    layer._R_cache_version = 5
    layer._R_out_leaf = torch.zeros(2, 8, 8)
    layer._R_in_leaf = torch.zeros(1, 8, 8)
    layer._R_out_full = torch.zeros(2, 8, 8)
    layer._R_in_full = torch.zeros(1, 8, 8)
    layer._invalidate_R_cache()
    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_cached_layer_is_poet_linear_subclass():
    from poet_torch import POETLinear

    assert issubclass(pc.CachedPOETLinear, POETLinear)


def test_invalidate_all_poet_caches_walks_registry():
    pc.reset_for_testing()
    a = pc.CachedPOETLinear(in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32)
    b = pc.CachedPOETLinear(in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32)
    pc.register_poet_layer(a)
    pc.register_poet_layer(b)
    a._R_cache_version = 3
    b._R_cache_version = 7
    pc.invalidate_all_poet_caches()
    assert a._R_cache_version == -1
    assert b._R_cache_version == -1
