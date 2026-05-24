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


def test_compute_cayley_matches_upstream_get_weight_poet():
    """_compute_cayley must produce the same (R_out, R_in) as the
    upstream get_weight_poet helper on identical inputs.

    GPU-only because torch.ops.poet.cayley is a Triton kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    from poet_torch.poet_layer import get_weight_poet

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=16,
        out_features=32,
        bsz=16,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    layer.random_init_parameters()

    R_out_ref, R_in_ref = get_weight_poet(  # noqa: N806
        layer.oft_R,
        layer.block_size,
        layer.rows,
        layer.cols,
        layer.r_out,
        layer.r_in,
    )
    R_out, R_in = pc._compute_cayley(  # noqa: N806
        layer.oft_R,
        layer.block_size,
        layer.rows,
        layer.cols,
        layer.r_in,
        layer.r_out,
    )
    assert torch.allclose(R_out, R_out_ref, atol=1e-6)
    assert torch.allclose(R_in, R_in_ref, atol=1e-6)


def test_forward_none_mode_matches_upstream_poet_linear():
    """`none` cache mode must produce the same output as upstream
    POETLinear.forward for the same inputs.

    GPU-only because the chain-layer kernel is a Triton kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    from poet_torch import POETLinear

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    torch.manual_seed(0)
    cached = pc.CachedPOETLinear(
        in_features=16,
        out_features=32,
        bsz=16,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    cached.random_init_parameters()

    torch.manual_seed(0)
    ref = POETLinear(
        in_features=16,
        out_features=32,
        bsz=16,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    ref.random_init_parameters()
    ref.weight.detach().copy_(cached.weight.detach())
    ref.oft_R.detach().copy_(cached.oft_R.detach())
    ref.perm_in.copy_(cached.perm_in)
    ref.perm_in_inv.copy_(cached.perm_in_inv)
    ref.perm_out.copy_(cached.perm_out)
    ref.perm_out_inv.copy_(cached.perm_out_inv)

    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    y_cached = cached(x)
    y_ref = ref(x)
    assert torch.allclose(y_cached, y_ref, atol=1e-5)
