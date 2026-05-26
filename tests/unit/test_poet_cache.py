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


def test_compute_cayley_decoupled_matches_upstream_helper():
    """_compute_cayley_decoupled must produce the same (R_out, R_in) as the
    upstream get_weight_poet_decoupled helper on identical inputs, with the
    block sizes threaded through correctly. Also checks near-orthogonality.

    GPU-only because torch.ops.poet.cayley is a Triton kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    from poet_torch.poet_layer import get_weight_poet_decoupled

    pc.reset_for_testing()
    # Decoupled layer: in=16, out=64, block_count=4 → bs_in=4, bs_out=16.
    layer = pc.CachedPOETLinear(
        in_features=16,
        out_features=64,
        block_count=4,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    layer.random_init_parameters()

    R_out_ref, R_in_ref = get_weight_poet_decoupled(  # noqa: N806
        layer.oft_R_in,
        layer.oft_R_out,
        layer.block_size_in,
        layer.block_size_out,
        layer.rows_in,
        layer.cols_in,
        layer.rows_out,
        layer.cols_out,
    )
    R_out, R_in = pc._compute_cayley_decoupled(  # noqa: N806
        layer.oft_R_in,
        layer.oft_R_out,
        layer.block_size_in,
        layer.block_size_out,
        layer.rows_in,
        layer.cols_in,
        layer.rows_out,
        layer.cols_out,
    )
    assert R_in.shape == (4, 4, 4)
    assert R_out.shape == (4, 16, 16)
    assert torch.allclose(R_out, R_out_ref, atol=1e-6)
    assert torch.allclose(R_in, R_in_ref, atol=1e-6)
    # Near-orthogonal (small init).
    eye_in = torch.eye(4, device="cuda").unsqueeze(0)
    assert (R_in @ R_in.transpose(-2, -1) - eye_in).abs().max() < 1e-3


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
    ref.oft_R_in.detach().copy_(cached.oft_R_in.detach())
    ref.oft_R_out.detach().copy_(cached.oft_R_out.detach())
    ref.perm_in.copy_(cached.perm_in)
    ref.perm_in_inv.copy_(cached.perm_in_inv)
    ref.perm_out.copy_(cached.perm_out)
    ref.perm_out_inv.copy_(cached.perm_out_inv)

    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    y_cached = cached(x)
    y_ref = ref(x)
    assert torch.allclose(y_cached, y_ref, atol=1e-5)


def _build_layer_for_parity(seed=0, dtype=torch.float32, device="cuda"):
    torch.manual_seed(seed)
    layer = pc.CachedPOETLinear(
        in_features=16,
        out_features=32,
        bsz=16,
        bias=False,
        device=device,
        dtype=dtype,
    )
    layer.random_init_parameters()
    layer.oft_R_in.requires_grad_(True)
    layer.oft_R_out.requires_grad_(True)
    return layer


def _stub_compute_cayley_decoupled(
    oft_in, oft_out, bs_in, bs_out, rows_in, cols_in, rows_out, cols_out
):
    """Stub mirroring _compute_cayley_decoupled's signature/return order.

    R_in depends only on oft_in and R_out only on oft_out, matching the real
    decoupled graph so the two-VJP flush is exercised. Called positionally.
    """
    r_in = oft_in.shape[0]
    r_out = oft_out.shape[0]
    eye_out = torch.eye(bs_out).unsqueeze(0).repeat(r_out, 1, 1) * oft_out.sum()
    eye_in = torch.eye(bs_in).unsqueeze(0).repeat(r_in, 1, 1) * oft_in.sum()
    return eye_out, eye_in


def test_mode_a_caches_cayley_across_K_calls(monkeypatch):  # noqa: N802
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    pc.register_poet_layer(layer)

    call_count = {"n": 0}

    def stub(*args, **kwargs):
        call_count["n"] += 1
        return _stub_compute_cayley_decoupled(*args, **kwargs)

    monkeypatch.setattr(pc, "_compute_cayley_decoupled", stub)

    for _ in range(4):
        layer._get_R_blocks_mode_a()
    assert call_count["n"] == 1
    assert layer._R_out_full is not None
    assert layer._R_in_full is not None
    assert layer._R_out_leaf.requires_grad
    assert layer._R_in_leaf.requires_grad


def test_mode_a_flush_writes_to_oft_R_grad_when_no_main_grad(monkeypatch):  # noqa: N802
    """Without Megatron's main_grad buffer, flush falls back to .grad
    so unit tests can exercise the flush math directly."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R_in.requires_grad_(True)
    layer.oft_R_out.requires_grad_(True)

    monkeypatch.setattr(pc, "_compute_cayley_decoupled", _stub_compute_cayley_decoupled)

    layer._get_R_blocks_mode_a()
    # Simulate K=2 micro-batch backwards depositing into R-leaf .grad.
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    # oft_R params have no main_grad attribute → flush writes to .grad on both.
    assert not hasattr(layer.oft_R_in, "main_grad")
    assert not hasattr(layer.oft_R_out, "main_grad")
    layer._flush_R_grads_to_oft_R()
    assert layer.oft_R_in.grad is not None and layer.oft_R_out.grad is not None
    assert torch.isfinite(layer.oft_R_in.grad).all()
    assert torch.isfinite(layer.oft_R_out.grad).all()


def test_mode_a_flush_writes_to_main_grad_when_present(monkeypatch):
    """When the parameter has a main_grad buffer (Megatron's FP32 grad
    accumulator), the flush writes there — not to .grad — so the outer
    optimizer's prepare_grads picks it up."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R_in.requires_grad_(True)
    layer.oft_R_out.requires_grad_(True)
    # Simulate Megatron's main_grad buffer (FP32 zero-initialized) on both params.
    layer.oft_R_in.main_grad = torch.zeros_like(layer.oft_R_in, dtype=torch.float32)
    layer.oft_R_out.main_grad = torch.zeros_like(layer.oft_R_out, dtype=torch.float32)

    monkeypatch.setattr(pc, "_compute_cayley_decoupled", _stub_compute_cayley_decoupled)

    layer._get_R_blocks_mode_a()
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    layer._flush_R_grads_to_oft_R()
    # main_grad must be populated on both; .grad must NOT (flush bypasses it).
    assert (layer.oft_R_in.main_grad != 0).any()
    assert (layer.oft_R_out.main_grad != 0).any()
    assert layer.oft_R_in.grad is None and layer.oft_R_out.grad is None


def test_mode_a_flush_invalidates_cache_after_running():
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R_in.requires_grad_(True)
    layer.oft_R_out.requires_grad_(True)
    layer._R_out_full = torch.zeros(2, 8, 8) * layer.oft_R_out.sum()
    layer._R_in_full = torch.zeros(1, 8, 8) * layer.oft_R_in.sum()
    layer._R_out_leaf = layer._R_out_full.detach().requires_grad_(True)
    layer._R_in_leaf = layer._R_in_full.detach().requires_grad_(True)
    layer._R_out_leaf.grad = torch.zeros_like(layer._R_out_leaf)
    layer._R_in_leaf.grad = torch.zeros_like(layer._R_in_leaf)
    layer._R_cache_version = 1

    layer._flush_R_grads_to_oft_R()
    assert layer._R_cache_version == -1
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_mode_a_flush_is_noop_when_no_forward_happened():
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._flush_R_grads_to_oft_R()  # must not raise
    assert layer.oft_R_in.grad is None and layer.oft_R_out.grad is None


def test_mode_a_K_microbatch_parity_with_none():  # noqa: N802
    """K=4 micro-batches: mode A's flushed grad must match mode none's
    accumulated oft_R.grad within float tolerance.

    This test runs the flush in isolation (no Megatron optimizer wrapper),
    so we check the `.grad` fallback path. The full pipeline behavior
    with main_grad is covered by the GPU smoke runbook (Task 11).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")

    K = 4  # noqa: N806
    xs = [torch.randn(4, 16, device="cuda", dtype=torch.float32) for _ in range(K)]

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    layer_n = _build_layer_for_parity()
    for x in xs:
        y = layer_n(x)
        y.sum().backward()
    g_n_in = layer_n.oft_R_in.grad.detach().clone()
    g_n_out = layer_n.oft_R_out.grad.detach().clone()

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    layer_a = _build_layer_for_parity()
    pc.register_poet_layer(layer_a)
    for x in xs:
        y = layer_a(x)
        y.sum().backward()
    layer_a._flush_R_grads_to_oft_R()
    g_a_in = layer_a.oft_R_in.grad.detach().clone()
    g_a_out = layer_a.oft_R_out.grad.detach().clone()

    # Parity is bit-exact at K=1; for K>1 the per-element rounding diff grows
    # only because grad magnitudes grow with K, while the *relative* error
    # stays at the fp32 floor. Assert a magnitude-aware relative error on each
    # of the two decoupled grads (a real logic bug would be >>1e-4).
    g_n = torch.cat([g_n_in.flatten(), g_n_out.flatten()])
    g_a = torch.cat([g_a_in.flatten(), g_a_out.flatten()])
    rel_err = (g_n - g_a).norm() / g_n.norm().clamp_min(1e-12)
    assert rel_err < 1e-4, f"mode A vs none relative grad error {rel_err:.2e} >= 1e-4"


def test_mode_a_K_microbatch_parity_decoupled():  # noqa: N802
    """Task 6.5: same K-microbatch parity, but on a DECOUPLED layer
    (in≠out, block_count=4 ⇒ block_size_in≠block_size_out). Mode A's flushed
    grad on both oft_R_in and oft_R_out must match none-mode accumulation."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")

    K = 4  # noqa: N806
    xs = [torch.randn(4, 32, device="cuda", dtype=torch.float32) for _ in range(K)]

    def build():
        torch.manual_seed(0)
        layer = pc.CachedPOETLinear(
            in_features=32,
            out_features=64,
            block_count=4,  # bs_in=8, bs_out=16
            bias=False,
            device="cuda",
            dtype=torch.float32,
        )
        layer.random_init_parameters()
        layer.oft_R_in.requires_grad_(True)
        layer.oft_R_out.requires_grad_(True)
        return layer

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    layer_n = build()
    for x in xs:
        layer_n(x).sum().backward()
    g_n = torch.cat([layer_n.oft_R_in.grad.flatten(), layer_n.oft_R_out.grad.flatten()])

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    layer_a = build()
    pc.register_poet_layer(layer_a)
    for x in xs:
        layer_a(x).sum().backward()
    layer_a._flush_R_grads_to_oft_R()
    g_a = torch.cat([layer_a.oft_R_in.grad.flatten(), layer_a.oft_R_out.grad.flatten()])

    rel_err = (g_n - g_a).norm() / g_n.norm().clamp_min(1e-12)
    assert rel_err < 1e-4, f"decoupled mode A vs none relative grad error {rel_err:.2e} >= 1e-4"


def test_sync_helper_is_safe_noop_on_cpu():
    """The DP sync helper must be safe to call on a CPU dev box (no
    Megatron, no torch.distributed init)."""
    from src.optim.poet import _sync_oft_R_grads_across_dp

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    # Decoupled layer carries two oft_R params; set main_grad on both.
    layer.oft_R_in.main_grad = torch.ones_like(layer.oft_R_in, dtype=torch.float32)
    layer.oft_R_out.main_grad = torch.ones_like(layer.oft_R_out, dtype=torch.float32) * 2
    snap_in = layer.oft_R_in.main_grad.clone()
    snap_out = layer.oft_R_out.main_grad.clone()

    _sync_oft_R_grads_across_dp([layer])
    # Single process → no op on either param.
    assert torch.equal(layer.oft_R_in.main_grad, snap_in)
    assert torch.equal(layer.oft_R_out.main_grad, snap_out)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2-rank DDP smoke requires 2 GPUs",
)
def test_mode_a_ddp_smoke_placeholder():
    """The full 2-rank DDP smoke must be driven via torchrun — see
    docs/superpowers/runbooks/2026-05-24-poet-cayley-cache-smoke.md.
    This placeholder exists to keep the test surface aware of it.
    """
    pytest.skip("Run via torchrun; see Task 11 runbook.")
