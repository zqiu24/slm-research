# ruff: noqa: N803, N806, RUF001, RUF002, RUF003, RUF005
"""Standalone benchmark for the POET Cayley cache (Mode A only).

One CachedPOETLinear, K micro-batches per cycle, comparing the cached path
against the uncached baseline.

- correctness: oft_R.grad max-abs and L2-relative error vs `none`,
  reported against the magnitude of the baseline grad so you can tell
  bf16 floor from a real bug.
- diagnostics: a K=1 control isolates the cache mechanism from
  multi-microbatch accumulation precision.
- speed: median wall-clock per gradient-accumulation cycle.

Usage:
    python tools/poet_cache_bench.py
    python tools/poet_cache_bench.py --dtype fp32      # precision control
    python tools/poet_cache_bench.py --sweep           # shape/K grid

No Megatron, no DDP, no training loop — just the layer + cache + autograd.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

# Sweep mode tests many (in, out, block) shapes in one process. Upstream's
# `forward_core` is `@torch.compile(fullgraph=True)`, and dynamo recompiles
# per distinct shape signature. Default recompile_limit is 8 — bump it so
# wide sweeps don't crash partway.
import torch._dynamo

torch._dynamo.config.recompile_limit = 256

_REPO = __file__.rsplit("/tools/", 1)[0]
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from poet_torch import POETLinear  # noqa: E402
from poet_torch.poet_layer import (  # noqa: E402
    chain_layer_x_checkpoint_mem_o2_decoupled,
    get_weight_poet_decoupled,
    pytorch_skew_symmetric,
)

from src.optim import poet_cache as pc  # noqa: E402

MODES = ("none", "cached_fwd_bwd")


def _block_kwargs(bsz, block_count):
    """POETLinear takes exactly one of bsz / block_count."""
    return {"block_count": block_count} if block_count is not None else {"bsz": bsz}


def _oft_params(layer):
    """The two decoupled skew params trained by POET."""
    return (layer.oft_R_in, layer.oft_R_out)


def _grad_vec(layer):
    """Concatenated fp32 grad over both oft_R params (for parity comparison)."""
    return torch.cat([p.grad.detach().float().flatten() for p in _oft_params(layer)])


def _args_divisor(args):
    """The dim divisor implied by the chosen block mode."""
    bc = getattr(args, "block_count", None)
    return bc if bc is not None else args.block_size


def _layer_desc(args):
    """Human-readable block configuration for headers."""
    bc = getattr(args, "block_count", None)
    if bc is not None:
        bsi = args.in_features // bc
        bso = args.out_features // bc
        return f"block_count={bc} (bs_in={bsi}, bs_out={bso})"
    return f"block_size={args.block_size}"


def build_layer(in_f, out_f, bsz, dtype, device, seed, block_count=None):
    torch.manual_seed(seed)
    layer = pc.CachedPOETLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        device=device,
        dtype=dtype,
        **_block_kwargs(bsz, block_count),
    )
    layer.random_init_parameters()
    for p in _oft_params(layer):
        p.requires_grad_(True)
    return layer


def zero_grads(layer):
    for p in _oft_params(layer):
        if p.grad is not None:
            p.grad.zero_()


def run_cycle(layer, xs, mode):
    """One cycle: K microbatches of forward+backward; Mode A flushes."""
    for x in xs:
        y = layer(x)
        y.sum().backward()
    if mode == "cached_fwd_bwd":
        layer._flush_R_grads_to_oft_R()


def parity_check(in_f, out_f, bsz, dtype, device, K, batch_shape, block_count=None):
    """Run one cycle in each mode with identical inputs; report:
    - max_abs:    max |g_cached - g_none|
    - rel_l2:     ||g_cached - g_none||_2 / ||g_none||_2
    - ref_max:    max |g_none|         (gives the scale to interpret max_abs)
    - ref_l2:     ||g_none||_2
    """
    torch.manual_seed(42)
    xs = [torch.randn(*batch_shape, in_f, device=device, dtype=dtype) for _ in range(K)]

    grads = {}
    for mode in MODES:
        pc.reset_for_testing()
        pc.set_cache_mode(mode)
        layer = build_layer(in_f, out_f, bsz, dtype, device, seed=0, block_count=block_count)
        run_cycle(layer, xs, mode)
        grads[mode] = _grad_vec(layer)

    ref = grads["none"]
    ref_max = ref.abs().max().item()
    ref_l2 = ref.norm().item()

    out = {
        "none": {
            "max_abs": 0.0,
            "rel_l2": 0.0,
            "ref_max": ref_max,
            "ref_l2": ref_l2,
        }
    }
    for mode in MODES:
        if mode == "none":
            continue
        diff = grads[mode] - ref
        max_abs = diff.abs().max().item()
        l2 = diff.norm().item()
        rel_l2 = l2 / max(ref_l2, 1e-12)
        out[mode] = {
            "max_abs": max_abs,
            "rel_l2": rel_l2,
            "ref_max": ref_max,
            "ref_l2": ref_l2,
        }
    return out


def time_mode(
    mode, in_f, out_f, bsz, dtype, device, K, cycles, batch_shape, warmup=5, block_count=None
):
    """Median ms per K-microbatch cycle for a single mode."""
    pc.reset_for_testing()
    pc.set_cache_mode(mode)
    layer = build_layer(in_f, out_f, bsz, dtype, device, seed=1, block_count=block_count)

    torch.manual_seed(43)
    xs = [torch.randn(*batch_shape, in_f, device=device, dtype=dtype) for _ in range(K)]

    for _ in range(warmup):
        zero_grads(layer)
        run_cycle(layer, xs, mode)
        pc.bump_poet_version()  # simulate optimizer.step()

    times = []
    for _ in range(cycles):
        zero_grads(layer)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_cycle(layer, xs, mode)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        pc.bump_poet_version()

    return times


def fmt_ms(t):
    return f"{t:7.2f}"


def _time_event(fn, repeats):
    """Time a no-arg callable using CUDA events; return list of ms."""
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def profile_components(
    in_f, out_f, bsz, dtype, device, K, batch_shape, warmup=8, repeats=50, block_count=None
):
    """Time each component of one microbatch + per-cycle extras separately.

    Returns a dict with:
      - t_baseline_micro:  median ms of one baseline (none mode) fwd+bwd
      - t_modea_hit_micro: median ms of one Mode A fwd+bwd with cache HIT
      - t_modea_miss_micro: median ms of one Mode A fwd+bwd with cache MISS
                            (i.e. first call of a cycle, includes building R_full)
      - t_flush_only:      median ms of just the _flush_R_grads_to_oft_R call
      - t_cayley_embedded: t_baseline_micro − t_modea_hit_micro
                            (Cayley fwd+bwd cost as it appears INSIDE baseline's
                            compiled forward_core, since the only difference
                            between the two microbatches is the Cayley work)
      - cayley_frac:       t_cayley_embedded / t_baseline_micro
      - theoretical_modea_cycle:  (K-1)*hit + miss + flush
      - theoretical_baseline_cycle: K * baseline
      - theoretical_speedup
    """
    torch.manual_seed(0)
    xs = [torch.randn(*batch_shape, in_f, device=device, dtype=dtype) for _ in range(K)]
    x_single = xs[0]

    # ---- 1) Baseline (none) per-microbatch ----
    pc.reset_for_testing()
    pc.set_cache_mode("none")
    layer_b = build_layer(in_f, out_f, bsz, dtype, device, seed=1, block_count=block_count)

    for _ in range(warmup):  # warm dynamo + first-call compile
        y = layer_b(x_single)
        y.sum().backward()
        zero_grads(layer_b)

    def _one_baseline_micro():
        y = layer_b(x_single)
        y.sum().backward()
        zero_grads(layer_b)

    t_baseline = statistics.median(_time_event(_one_baseline_micro, repeats))

    # ---- 2) Mode A per-microbatch ON CACHE HIT ----
    # Build a fresh layer for Mode A.
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    layer_a = build_layer(in_f, out_f, bsz, dtype, device, seed=1, block_count=block_count)
    pc.register_poet_layer(layer_a)

    # Warm the cache: first call builds R_full + dynamo compiles _cached_chain_layer_core.
    # Subsequent calls hit the cache and the compile cache.
    for _ in range(warmup):
        y = layer_a(x_single)
        y.sum().backward()
        # leave R_leaf.grad accumulating — for timing it doesn't matter
        # (in-place float add cost is independent of value).

    def _one_modea_hit():
        y = layer_a(x_single)
        y.sum().backward()

    t_modea_hit = statistics.median(_time_event(_one_modea_hit, repeats))

    # ---- 3) Mode A CACHE MISS (first call after version bump) ----
    def _one_modea_miss():
        pc.bump_poet_version()  # forces miss on next forward
        layer_a._invalidate_R_cache()  # belt-and-suspenders
        y = layer_a(x_single)
        y.sum().backward()

    # Warm again (one miss-then-hit cycle) before timing miss
    for _ in range(3):
        _one_modea_miss()

    t_modea_miss = statistics.median(_time_event(_one_modea_miss, repeats))

    # ---- 4) Flush only ----
    def _one_flush():
        pc.bump_poet_version()
        layer_a._invalidate_R_cache()
        for x in xs:
            y = layer_a(x)
            y.sum().backward()
        # Now time only the flush. We need to wrap event recording around just the flush.
        # We'll do that in _time_event-like inner timing.

    # Custom timing — only the flush call is timed.
    flush_times = []
    for _ in range(repeats):
        pc.bump_poet_version()
        layer_a._invalidate_R_cache()
        for x in xs:
            y = layer_a(x)
            y.sum().backward()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        layer_a._flush_R_grads_to_oft_R()
        e.record()
        torch.cuda.synchronize()
        flush_times.append(s.elapsed_time(e))
    t_flush = statistics.median(flush_times)

    cayley_embed = max(0.0, t_baseline - t_modea_hit)
    cayley_frac = cayley_embed / t_baseline if t_baseline > 0 else 0.0

    theo_baseline = K * t_baseline
    theo_modea = (K - 1) * t_modea_hit + t_modea_miss + t_flush
    theo_speedup = theo_baseline / theo_modea if theo_modea > 0 else 0.0

    return {
        "t_baseline_micro": t_baseline,
        "t_modea_hit_micro": t_modea_hit,
        "t_modea_miss_micro": t_modea_miss,
        "t_flush_only": t_flush,
        "t_cayley_embedded": cayley_embed,
        "cayley_frac": cayley_frac,
        "theo_baseline_cycle": theo_baseline,
        "theo_modea_cycle": theo_modea,
        "theo_speedup": theo_speedup,
    }


def micro_profile(
    in_f, out_f, bsz, dtype, device, batch_shape, warmup=10, repeats=80, block_count=None
):
    """Break down a SINGLE upstream POETLinear microbatch into sub-operations.

    Goal: identify where time goes inside the baseline (uncached) POET layer,
    so we know which components are worth optimizing.

    Times these operations in isolation (no_grad unless noted):
      - pytorch_skew_symmetric            (Cayley step 1)
      - torch.ops.poet.cayley             (Cayley step 2 — Triton kernel)
      - _compute_cayley full              (Cayley fwd, end-to-end)
      - chain_layer_x_checkpoint_mem_o2   (linear+perms, current default)
      - chain_layer_x_checkpoint          (linear without perms — reference for perm cost)
      - upstream POETLinear forward       (compiled forward_core, fused)
      - upstream POETLinear fwd+bwd       (with autograd)
    """
    torch.manual_seed(0)
    layer = POETLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        device=device,
        dtype=dtype,
        **_block_kwargs(bsz, block_count),
    )
    layer.random_init_parameters()
    for p in (layer.oft_R_in, layer.oft_R_out):
        p.requires_grad_(True)

    x = torch.randn(*batch_shape, in_f, device=device, dtype=dtype)

    # Pre-compute R blocks once (for chain_layer-only timing)
    with torch.no_grad():
        R_out_pre, R_in_pre = get_weight_poet_decoupled(
            layer.oft_R_in,
            layer.oft_R_out,
            layer.block_size_in,
            layer.block_size_out,
            layer.rows_in,
            layer.cols_in,
            layer.rows_out,
            layer.cols_out,
        )

    def time_fn(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return statistics.median(times)

    # 1. pytorch_skew_symmetric (decoupled = two skews, one per side)
    def fn_skew():
        with torch.no_grad():
            _ = pytorch_skew_symmetric(
                layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in
            )
            _ = pytorch_skew_symmetric(
                layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out
            )

    t_skew = time_fn(fn_skew)

    # 2. torch.ops.poet.cayley (the Triton kernel) — both sides
    with torch.no_grad():
        Q_in_pre = pytorch_skew_symmetric(
            layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in
        )
        Q_out_pre = pytorch_skew_symmetric(
            layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out
        )

    def fn_cayley_kernel():
        with torch.no_grad():
            _ = torch.ops.poet.cayley(Q_in_pre)[0]
            _ = torch.ops.poet.cayley(Q_out_pre)[0]

    t_cayley_kernel = time_fn(fn_cayley_kernel)

    # 3. get_weight_poet_decoupled end-to-end (= two skews + two kernels)
    def fn_cayley_full():
        with torch.no_grad():
            _ = get_weight_poet_decoupled(
                layer.oft_R_in,
                layer.oft_R_out,
                layer.block_size_in,
                layer.block_size_out,
                layer.rows_in,
                layer.cols_in,
                layer.rows_out,
                layer.cols_out,
            )

    t_cayley_full = time_fn(fn_cayley_full)

    # 4. chain_layer with perms — EAGER (decoupled op)
    def fn_chain_with_perms_eager():
        with torch.no_grad():
            _ = chain_layer_x_checkpoint_mem_o2_decoupled(
                x,
                R_in_pre,
                layer.weight,
                layer.bias,
                R_out_pre,
                layer.perm_in_inv,
                layer.perm_in,
                layer.perm_out,
                layer.perm_out_inv,
                layer.block_size_in,
                layer.block_size_out,
            )

    t_chain_with_perms = time_fn(fn_chain_with_perms_eager)

    # 4b. chain_layer with perms — COMPILED (matches Mode A's cached core)
    @torch.compile(fullgraph=True)
    def _compiled_chain_with_perms(x, R_in, W, b, R_out, p_ii, p_i, p_o, p_oi, bs_in, bs_out):
        return chain_layer_x_checkpoint_mem_o2_decoupled(
            x,
            R_in,
            W,
            b,
            R_out,
            p_ii,
            p_i,
            p_o,
            p_oi,
            bs_in,
            bs_out,
        )

    def fn_chain_with_perms_compiled():
        with torch.no_grad():
            _ = _compiled_chain_with_perms(
                x,
                R_in_pre,
                layer.weight,
                layer.bias,
                R_out_pre,
                layer.perm_in_inv,
                layer.perm_in,
                layer.perm_out,
                layer.perm_out_inv,
                layer.block_size_in,
                layer.block_size_out,
            )

    t_chain_with_perms_compiled = time_fn(fn_chain_with_perms_compiled)

    # 5. chain_layer WITHOUT perms — NOT AVAILABLE in this build.
    # The op `poet::chain_layer_checkpoint` is defined in poet_ops.py but
    # not registered (the q8 variants are). We report perm overhead as
    # unavailable; the equivalent measurement via Path 2 would need either
    # rebuilding poet_torch or implementing a manual permute+matmul path.
    t_chain_no_perms = float("nan")
    t_chain_no_perms_compiled = float("nan")

    # 6. Upstream POETLinear forward only (compiled forward_core)
    def fn_layer_fwd():
        with torch.no_grad():
            _ = layer(x)

    t_layer_fwd = time_fn(fn_layer_fwd)

    # 7. Upstream POETLinear fwd + bwd
    def fn_layer_fwdbwd():
        y = layer(x)
        y.sum().backward()
        zero_grads(layer)

    t_layer_fwdbwd = time_fn(fn_layer_fwdbwd)

    # Derived
    import math

    t_layer_bwd = max(0.0, t_layer_fwdbwd - t_layer_fwd)
    perm_overhead_eager = (
        max(0.0, t_chain_with_perms - t_chain_no_perms)
        if not math.isnan(t_chain_no_perms)
        else float("nan")
    )
    perm_overhead_compiled = (
        max(0.0, t_chain_with_perms_compiled - t_chain_no_perms_compiled)
        if not math.isnan(t_chain_no_perms_compiled)
        else float("nan")
    )
    compile_savings = max(0.0, t_chain_with_perms - t_chain_with_perms_compiled)
    fusion_savings = max(0.0, (t_cayley_full + t_chain_with_perms_compiled) - t_layer_fwd)

    return {
        "t_skew": t_skew,
        "t_cayley_kernel": t_cayley_kernel,
        "t_cayley_full": t_cayley_full,
        "t_chain_with_perms": t_chain_with_perms,
        "t_chain_with_perms_compiled": t_chain_with_perms_compiled,
        "t_chain_no_perms": t_chain_no_perms,
        "t_chain_no_perms_compiled": t_chain_no_perms_compiled,
        "t_layer_fwd": t_layer_fwd,
        "t_layer_fwdbwd": t_layer_fwdbwd,
        "t_layer_bwd": t_layer_bwd,
        "perm_overhead_eager": perm_overhead_eager,
        "perm_overhead_compiled": perm_overhead_compiled,
        "compile_savings": compile_savings,
        "fusion_savings": fusion_savings,
    }


def print_micro_profile(args, dtype, device):
    print("Original POET layer micro-profile")
    print("=" * 60)
    print(f"  Layer:       in={args.in_features}  out={args.out_features}  {_layer_desc(args)}")
    print(f"  Input shape: {tuple(args.batch)} + (in,) = {tuple(args.batch) + (args.in_features,)}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print()

    div = _args_divisor(args)
    if args.in_features % div != 0 or args.out_features % div != 0:
        print(f"SKIP: {div} doesn't divide both dims")
        return

    p = micro_profile(
        args.in_features,
        args.out_features,
        args.block_size,
        dtype,
        device,
        args.batch,
        block_count=args.block_count,
    )

    fwd_total = p["t_layer_fwd"]
    full_total = p["t_layer_fwdbwd"]

    def pct(part, whole):
        return f"{100 * part / whole:5.1f}%" if whole > 0 else "  N/A"

    print("Forward (no-grad, single microbatch):")
    print("  Cayley fwd (eager):")
    print(f"    pytorch_skew_symmetric:                {p['t_skew']:7.3f} ms")
    print(f"    torch.ops.poet.cayley (Triton kernel): {p['t_cayley_kernel']:7.3f} ms")
    print(f"    _compute_cayley total:                 {p['t_cayley_full']:7.3f} ms")
    print()
    import math

    nan = math.isnan

    def fmt(v):
        return "    n/a" if nan(v) else f"{v:7.3f}"

    print("  chain_layer fwd:                            eager   |  compiled")
    print(
        f"    with perms (chain_layer_x_checkpoint_mem_o2): "
        f"{fmt(p['t_chain_with_perms'])} | {fmt(p['t_chain_with_perms_compiled'])} ms"
    )
    print(
        f"    no perms (chain_layer_x_checkpoint):          "
        f"{fmt(p['t_chain_no_perms'])} | {fmt(p['t_chain_no_perms_compiled'])} ms"
        f"  (op not registered in this build)"
    )
    print(
        f"    perm overhead (with − without):               "
        f"{fmt(p['perm_overhead_eager'])} | {fmt(p['perm_overhead_compiled'])} ms"
    )
    print(f"    @torch.compile savings (with perms):          {p['compile_savings']:7.3f} ms")
    print()
    print(
        f"  Compiled fused forward (upstream forward_core): {p['t_layer_fwd']:7.3f} ms  ← what baseline runs"
    )
    print(
        f"  Sum (Cayley + compiled chain w/ perms):         "
        f"{p['t_cayley_full'] + p['t_chain_with_perms_compiled']:7.3f} ms"
    )
    print(f"  Fusion savings (sum − compiled forward):        {p['fusion_savings']:7.3f} ms")
    print()

    print("Forward + backward (with autograd):")
    print(f"  Total fwd+bwd:                {full_total:7.3f} ms")
    print(
        f"  Forward (compiled):           {p['t_layer_fwd']:7.3f} ms  {pct(p['t_layer_fwd'], full_total)}"
    )
    print(
        f"  Backward (inferred):          {p['t_layer_bwd']:7.3f} ms  {pct(p['t_layer_bwd'], full_total)}"
    )
    print()

    cayley_frac_fwd = p["t_cayley_full"] / fwd_total if fwd_total > 0 else 0
    chain_frac_fwd = p["t_chain_with_perms_compiled"] / fwd_total if fwd_total > 0 else 0
    print("Fractions (relative to compiled forward):")
    print(f"  Cayley fwd / compiled fwd:           {100 * cayley_frac_fwd:5.1f}%")
    print(f"  chain_layer (compiled) / compiled fwd: {100 * chain_frac_fwd:5.1f}%")
    if p["t_chain_with_perms_compiled"] > 0 and not nan(p["perm_overhead_compiled"]):
        print(
            f"  Perm overhead / chain_layer (compiled): {100 * p['perm_overhead_compiled'] / p['t_chain_with_perms_compiled']:5.1f}%"
        )
    print()
    print("Bottleneck candidate ranking (by compiled component cost, fwd-only):")
    components = [
        ("chain_layer (with perms, compiled)", p["t_chain_with_perms_compiled"]),
        ("Cayley fwd (eager — never compiled)", p["t_cayley_full"]),
    ]
    if not nan(p["perm_overhead_compiled"]):
        components.append(("perm overhead (compiled)", p["perm_overhead_compiled"]))
    components_sorted = sorted(components, key=lambda c: -c[1])
    for label, t in components_sorted:
        print(f"  {label:<40s}{t:7.3f} ms")


def print_profile(args, dtype, device):
    print("POET Cayley cache component profile")
    print("=" * 60)
    print(f"  Layer:       in={args.in_features}  out={args.out_features}  {_layer_desc(args)}")
    print(f"  Input shape: {tuple(args.batch)} + (in,) = {tuple(args.batch) + (args.in_features,)}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  K (μbatches/cycle): {args.K}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print()

    div = _args_divisor(args)
    if args.in_features % div != 0 or args.out_features % div != 0:
        print(f"SKIP: {div} doesn't divide both dims")
        return

    p = profile_components(
        args.in_features,
        args.out_features,
        args.block_size,
        dtype,
        device,
        args.K,
        args.batch,
        block_count=args.block_count,
    )

    print("Per-microbatch / per-cycle component times (ms):")
    print(
        f"  baseline microbatch (fwd+bwd, compiled, includes Cayley): {p['t_baseline_micro']:.3f}"
    )
    print(
        f"  Mode A microbatch on cache HIT  (fwd+bwd, compiled, no Cayley): {p['t_modea_hit_micro']:.3f}"
    )
    print(
        f"  Mode A microbatch on cache MISS (fwd+bwd, builds R_full): {p['t_modea_miss_micro']:.3f}"
    )
    print(f"  Mode A flush only (1 Cayley bwd via autograd.grad): {p['t_flush_only']:.3f}")
    print()
    print("Derived:")
    print(
        f"  Cayley embedded in baseline microbatch: {p['t_cayley_embedded']:.3f} ms"
        f"  (= baseline − Mode A hit)"
    )
    print(f"  Cayley fraction of baseline step:       {p['cayley_frac'] * 100:.1f}%")
    print(
        f"  Cache-miss extra cost (vs hit):         {p['t_modea_miss_micro'] - p['t_modea_hit_micro']:.3f} ms"
    )
    print()
    print(f"Theoretical per-cycle (K={args.K}):")
    print(f"  baseline:  K × baseline_micro          = {p['theo_baseline_cycle']:.2f} ms")
    print(f"  Mode A:    (K−1)×hit + miss + flush    = {p['theo_modea_cycle']:.2f} ms")
    print(f"  Speedup (theory from components):       {p['theo_speedup']:.3f}x")
    print()

    # Compare to a full end-to-end measurement.
    print("End-to-end measurement (same point, single-mode timing):")
    pc.reset_for_testing()
    pc.set_cache_mode("none")
    baseline_full = time_mode(
        "none",
        args.in_features,
        args.out_features,
        args.block_size,
        dtype,
        device,
        args.K,
        args.cycles,
        args.batch,
        block_count=args.block_count,
    )
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    modea_full = time_mode(
        "cached_fwd_bwd",
        args.in_features,
        args.out_features,
        args.block_size,
        dtype,
        device,
        args.K,
        args.cycles,
        args.batch,
        block_count=args.block_count,
    )
    med_b = statistics.median(baseline_full)
    med_a = statistics.median(modea_full)
    print(f"  baseline cycle (measured):    {med_b:.2f} ms")
    print(f"  Mode A cycle (measured):      {med_a:.2f} ms")
    print(f"  Speedup (measured):           {med_b / med_a:.3f}x")
    print()
    print("Discrepancy between theory and measurement:")
    print(
        f"  Mode A cycle: theory {p['theo_modea_cycle']:.2f} vs measured {med_a:.2f}"
        f"  → overhead = {med_a - p['theo_modea_cycle']:.2f} ms"
        f" ({100 * (med_a - p['theo_modea_cycle']) / med_a:+.1f}%)"
    )


def run_point(in_f, out_f, bsz, K, cycles, dtype, device, batch_shape, warmup=5, block_count=None):
    divisor = block_count if block_count is not None else bsz
    if in_f % divisor != 0 or out_f % divisor != 0:
        return None
    parity = parity_check(in_f, out_f, bsz, dtype, device, K, batch_shape, block_count=block_count)
    timings = {
        m: time_mode(
            m,
            in_f,
            out_f,
            bsz,
            dtype,
            device,
            K,
            cycles,
            batch_shape,
            warmup,
            block_count=block_count,
        )
        for m in MODES
    }
    return parity, timings


def print_single(args, dtype, device):
    print("POET Cayley cache benchmark (Mode A)")
    print("=" * 60)
    print(f"  Layer:       in={args.in_features}  out={args.out_features}  {_layer_desc(args)}")
    print(f"  Input shape: {tuple(args.batch)} + (in,) = {tuple(args.batch) + (args.in_features,)}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  K (μbatches/cycle): {args.K}")
    print(f"  Timing cycles:      {args.cycles}  (after {args.warmup} warmup cycles)")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print()

    out = run_point(
        args.in_features,
        args.out_features,
        args.block_size,
        args.K,
        args.cycles,
        dtype,
        device,
        args.batch,
        warmup=args.warmup,
        block_count=args.block_count,
    )
    if out is None:
        print(
            f"SKIP: {_args_divisor(args)} does not divide both "
            f"{args.in_features} and {args.out_features}"
        )
        return
    parity, timings = out

    print("Correctness: oft_R.grad vs `none`")
    print(f"  {'mode':<18}{'max abs':>13}{'rel L2':>13}{'ref max':>13}{'ref L2':>13}")
    for mode in MODES:
        m = parity[mode]
        print(
            f"  {mode:<18}{m['max_abs']:>13.3e}{m['rel_l2']:>13.3e}"
            f"{m['ref_max']:>13.3e}{m['ref_l2']:>13.3e}"
        )
    print()

    print("Speed: median ms per gradient-accumulation cycle")
    med_none = statistics.median(timings["none"])
    print(f"  {'mode':<18}{'median ms':>12}{'p5':>10}{'p95':>10}{'speedup':>12}")
    for mode in MODES:
        ts = sorted(timings[mode])
        med = statistics.median(ts)
        p5 = ts[max(0, int(0.05 * len(ts)))]
        p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
        speedup = med_none / med
        print(f"  {mode:<18}{fmt_ms(med):>12}{fmt_ms(p5):>10}{fmt_ms(p95):>10}{speedup:>11.2f}x")
    print()

    sA = med_none / statistics.median(timings["cached_fwd_bwd"])
    if sA > 1 and args.K > 1:
        cayley_frac = (1.0 - 1.0 / sA) * args.K / (args.K - 1)
        print(f"  Implied Cayley fraction of step time: {cayley_frac * 100:5.1f}%")


# Default sweep grid. Pruned to fit upstream POETLinear's torch.compile
# recompile_limit of 8 distinct (in, out, block) shape signatures per
# process: each sweep run picks at most 8 unique shape×block combos.
DEFAULT_SHAPES = [
    (1536, 1536, "llama3_1.2b_qkv"),
    (1536, 3840, "llama3_1.2b_ffn_up"),
    (3840, 1536, "llama3_1.2b_ffn_dn"),
    (4096, 4096, "qwen3_7b_qkv"),
    (4096, 11008, "qwen3_7b_ffn_up"),
    (11008, 4096, "qwen3_7b_ffn_dn"),
    (7168, 7168, "kimi_k2_qkv"),
]
DEFAULT_BLOCK_SIZES = [256]
DEFAULT_BLOCK_COUNTS = [4, 8, 16, 32]
DEFAULT_KS = [8, 16, 32, 64]


def print_sweep(args, dtype, device):
    shapes = (
        DEFAULT_SHAPES
        if args.shapes is None
        else [(int(a), int(b), f"{a}x{b}") for a, b in (s.split("x") for s in args.shapes)]
    )
    block_sizes = args.block_sizes or DEFAULT_BLOCK_SIZES
    block_counts = args.block_counts or []
    Ks = args.Ks or DEFAULT_KS

    # Unified block-knob list: ("size", N) sweeps a shared block size;
    # ("count", N) sweeps decoupled block_count (bs_in=in/N, bs_out=out/N).
    specs = [("size", b) for b in block_sizes] + [("count", c) for c in block_counts]

    print("POET Cayley cache sweep (Mode A)")
    print("=" * 60)
    print(f"  Shapes:      {len(shapes)}")
    for in_f, out_f, label in shapes:
        print(f"    {label:<22} ({in_f:>5} × {out_f:>5})")
    print(f"  Block sizes:  {block_sizes}")
    print(f"  Block counts: {block_counts}")
    print(f"  K values:    {Ks}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Input shape: {tuple(args.batch)} + (in,)")
    print(f"  Cycles:      {args.cycles} timed, {args.warmup} warmup")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print()

    hdr = (
        f"{'shape':<22}{'in':>6}{'out':>7}{'block':>10}{'K':>4}"
        f"{'err_max':>11}{'rel_L2':>11}{'ref_max':>11}"
        f"{'ms_none':>10}{'ms_A':>10}{'sp_A':>8}{'cayley%':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for in_f, out_f, label in shapes:
        for kind, n in specs:
            # block label + the (bsz, block_count) args for this point.
            if kind == "size":
                if in_f % n != 0 or out_f % n != 0:
                    continue
                blk_label = f"bs{n}"
                bsz, block_count = n, None
            else:  # count
                if in_f % n != 0 or out_f % n != 0:
                    continue
                blk_label = f"bc{n}"
                bsz, block_count = 256, n
            for K in Ks:
                out = run_point(
                    in_f,
                    out_f,
                    bsz,
                    K,
                    args.cycles,
                    dtype,
                    device,
                    args.batch,
                    warmup=args.warmup,
                    block_count=block_count,
                )
                if out is None:
                    continue
                parity, timings = out
                med_none = statistics.median(timings["none"])
                med_A = statistics.median(timings["cached_fwd_bwd"])
                sp_A = med_none / med_A
                err_max = parity["cached_fwd_bwd"]["max_abs"]
                rel_l2 = parity["cached_fwd_bwd"]["rel_l2"]
                ref_max = parity["cached_fwd_bwd"]["ref_max"]
                cayley_frac = (
                    (1.0 - 1.0 / sp_A) * K / (K - 1) * 100.0 if sp_A > 1.0 and K > 1 else 0.0
                )
                row = (
                    f"{label:<22}{in_f:>6}{out_f:>7}{blk_label:>10}{K:>4}"
                    f"{err_max:>11.2e}{rel_l2:>11.2e}{ref_max:>11.2e}"
                    f"{med_none:>10.2f}{med_A:>10.2f}{sp_A:>7.2f}x{cayley_frac:>8.1f}%"
                )
                print(row, flush=True)
                rows.append(row)

    print()
    print(f"  Total rows: {len(rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-features", type=int, default=1536)
    p.add_argument("--out-features", type=int, default=1536)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument(
        "--block-count",
        type=int,
        default=None,
        help="decoupled mode: each side gets N blocks (bs_in=in/N, bs_out=out/N). "
        "Mutually exclusive with --block-size; takes precedence when set.",
    )
    p.add_argument("--K", type=int, default=16, help="microbatches per cycle")
    p.add_argument("--cycles", type=int, default=50, help="cycles to time")
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warmup cycles before timing (discarded; warms caches + torch.compile)",
    )
    p.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[32, 256],
        help="leading dims before in_features (e.g. 32 256 → x.shape=(32,256,in))",
    )
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--sweep", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="time each component (baseline micro / Mode A hit / miss / flush) separately",
    )
    p.add_argument(
        "--micro-profile",
        action="store_true",
        help="break down the original POET layer's forward + backward into sub-operations",
    )
    p.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help='shapes for sweep, e.g. "1536x1536 1536x3840" (default: pruned to ≤8 to stay under torch.compile recompile_limit)',
    )
    p.add_argument("--block-sizes", type=int, nargs="+", default=None)
    p.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=None,
        help=f"sweep these decoupled block_counts in addition to --block-sizes "
        f"(suggested: {DEFAULT_BLOCK_COUNTS}). Invalid (non-dividing) combos are skipped.",
    )
    p.add_argument("--Ks", type=int, nargs="+", default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required (Triton kernels are GPU-only).")
        sys.exit(1)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"

    if args.micro_profile:
        print_micro_profile(args, dtype, device)
    elif args.profile:
        print_profile(args, dtype, device)
    elif args.sweep:
        print_sweep(args, dtype, device)
    else:
        print_single(args, dtype, device)


if __name__ == "__main__":
    main()
