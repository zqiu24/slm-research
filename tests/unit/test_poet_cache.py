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


def test_mode_b_caches_cayley_across_K_calls(monkeypatch):  # noqa: N802
    """In cached_fwd mode, _compute_cayley runs once per cache version,
    not K times across K forward calls in the same accumulation cycle."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")

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

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):  # noqa: N803
        call_count["n"] += 1
        R_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)  # noqa: N806
        R_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)  # noqa: N806
        return R_out, R_in

    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    for _ in range(4):
        _R_out, _R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)  # noqa: N806
    assert call_count["n"] == 1

    pc.bump_poet_version()
    _R_out, _R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)  # noqa: N806
    assert call_count["n"] == 2


def test_mode_b_backward_runs_cayley_K_times(monkeypatch):  # noqa: N802
    """Mode B's backward rebuilds the cayley graph on every call —
    confirms the K→1 saving is on the forward only."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)

    call_count = {"n": 0}

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):  # noqa: N803
        call_count["n"] += 1
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(2, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(1, 1, 1)
        return eye_out * scale, eye_in * scale

    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    for _ in range(3):
        R_out, R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)  # noqa: N806
        (R_out.sum() + R_in.sum()).backward()
    # 1 forward (first call only) + 3 backwards = 4 calls.
    assert call_count["n"] == 4


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
    layer.oft_R.requires_grad_(True)
    return layer


def test_mode_b_single_microbatch_parity_with_none():
    """Mode B's forward output and oft_R.grad must match mode none
    within float tolerance for a single forward+backward.

    GPU-only.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")

    pc.reset_for_testing()
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)

    pc.set_cache_mode("none")
    layer_n = _build_layer_for_parity()
    y_n = layer_n(x)
    y_n.sum().backward()
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.set_cache_mode("cached_fwd")
    layer_b = _build_layer_for_parity()
    y_b = layer_b(x)
    y_b.sum().backward()
    g_b = layer_b.oft_R.grad.detach().clone()

    assert torch.allclose(y_n, y_b, atol=1e-5)
    assert torch.allclose(g_n, g_b, atol=1e-5)


def test_mode_b_K_microbatch_accumulation_parity_with_none():  # noqa: N802
    """K=4 micro-batches: mode B's accumulated oft_R.grad must match
    mode none within float tolerance. Spec §13.2.

    GPU-only.
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
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")
    layer_b = _build_layer_for_parity()
    for x in xs:
        y = layer_b(x)
        y.sum().backward()
    g_b = layer_b.oft_R.grad.detach().clone()

    assert torch.allclose(g_n, g_b, atol=1e-5)


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

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):  # noqa: N803
        call_count["n"] += 1
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale

    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

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
    layer.oft_R.requires_grad_(True)

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):  # noqa: N803
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale

    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    layer._get_R_blocks_mode_a()
    # Simulate K=2 micro-batch backwards depositing into R-leaf .grad.
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    # oft_R has no main_grad attribute → flush writes to .grad.
    assert not hasattr(layer.oft_R, "main_grad")
    layer._flush_R_grads_to_oft_R()
    assert layer.oft_R.grad is not None
    assert torch.isfinite(layer.oft_R.grad).all()


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
    layer.oft_R.requires_grad_(True)
    # Simulate Megatron's main_grad buffer (FP32 zero-initialized).
    layer.oft_R.main_grad = torch.zeros_like(layer.oft_R, dtype=torch.float32)

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):  # noqa: N803
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale

    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    layer._get_R_blocks_mode_a()
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    layer._flush_R_grads_to_oft_R()
    # main_grad must be populated; .grad must NOT (the flush bypasses it).
    assert (layer.oft_R.main_grad != 0).any()
    assert layer.oft_R.grad is None


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
    layer.oft_R.requires_grad_(True)
    layer._R_out_full = torch.zeros(2, 8, 8) * layer.oft_R.sum()
    layer._R_in_full = torch.zeros(1, 8, 8) * layer.oft_R.sum()
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
    assert layer.oft_R.grad is None


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
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    layer_a = _build_layer_for_parity()
    pc.register_poet_layer(layer_a)
    for x in xs:
        y = layer_a(x)
        y.sum().backward()
    layer_a._flush_R_grads_to_oft_R()
    g_a = layer_a.oft_R.grad.detach().clone()

    # Parity is bit-exact at K=1; for K>1 the per-element rounding diff grows
    # only because grad magnitudes grow with K, while the *relative* error
    # stays at the fp32 floor (~1.7e-6 measured on B200). With grads here of
    # O(10-50), the plan's original atol=1e-5 demanded ~3e-7 relative
    # precision, which fp32 Triton kernels cannot deliver. Assert a
    # magnitude-aware relative error instead (a real logic bug would be >>1e-4).
    rel_err = (g_n - g_a).norm() / g_n.norm().clamp_min(1e-12)
    assert rel_err < 1e-4, f"mode A vs none relative grad error {rel_err:.2e} >= 1e-4"


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
    layer.oft_R.main_grad = torch.ones_like(layer.oft_R, dtype=torch.float32)
    snapshot = layer.oft_R.main_grad.clone()

    _sync_oft_R_grads_across_dp([layer])
    # Single process → no op.
    assert torch.equal(layer.oft_R.main_grad, snapshot)


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
