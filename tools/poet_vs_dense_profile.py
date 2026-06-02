# ruff: noqa
"""Single-GPU kernel-level breakdown of POET overhead vs a plain (Adam) linear.

Builds one 300m transformer layer's worth of linears (q,k,v,o,gate,up,down) as
POETLinear (block_size=256, fast non-recompute path) and as plain nn.Linear
(trainable weight = Adam baseline), runs fwd+bwd at the real token count, and
reports:

  * end-to-end CUDA time for POET vs dense (the per-layer analogue of the 2x),
  * a per-op CUDA-time breakdown for POET grouped into
      mm (dense matmul) | bmm (block rotations) | cayley | gather/index | other

so we can see exactly which kernels eat the extra time.

Run (single GPU, same venv as training):
    CUDA_VISIBLE_DEVICES=0 python tools/poet_vs_dense_profile.py
    CUDA_VISIBLE_DEVICES=0 python tools/poet_vs_dense_profile.py --block-size 256 --tokens 32768 --reps 20
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile

_REPO = __file__.rsplit("/tools/", 1)[0]
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from poet_torch import POETLinear  # noqa: E402
from poet_torch.poet_layer import (  # noqa: E402
    chain_layer_x_fast_decoupled,
    get_weight_poet_decoupled,
)

# 300m llama3 layer (hidden=1024, ffn=2560, GQA q=1024 / kv=256).
LINEARS = [
    ("q", 1024, 1024),
    ("k", 1024, 256),
    ("v", 1024, 256),
    ("o", 1024, 1024),
    ("gate", 1024, 2560),
    ("up", 1024, 2560),
    ("down", 2560, 1024),
]


def _evt():
    return torch.cuda.Event(enable_timing=True)


def build(block_size, dtype, dev):
    poet, dense, inputs = [], [], []
    for name, i, o in LINEARS:
        pl = POETLinear(i, o, bsz=block_size, bias=False, device=dev, dtype=dtype)
        # non-zero oft_R so Cayley does representative work (not identity).
        with torch.no_grad():
            pl.oft_R_in.normal_(0, 1e-2)
            pl.oft_R_out.normal_(0, 1e-2)
        poet.append(pl)
        dl = torch.nn.Linear(i, o, bias=False, device=dev, dtype=dtype)  # weight trainable -> Adam
        dense.append(dl)
        inputs.append(i)
    return poet, dense, inputs


def fwd_bwd(modules, in_dims, tokens, dtype, dev):
    for m, i in zip(modules, in_dims):
        x = torch.randn(tokens, i, device=dev, dtype=dtype, requires_grad=True)
        y = m(x)
        y.sum().backward()


def timed(modules, in_dims, tokens, dtype, dev, reps):
    for _ in range(3):  # warmup / compile
        fwd_bwd(modules, in_dims, tokens, dtype, dev)
    torch.cuda.synchronize()
    s, e = _evt(), _evt()
    ts = []
    for _ in range(reps):
        s.record()
        fwd_bwd(modules, in_dims, tokens, dtype, dev)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def precompute_R(poet):
    """One R per layer (skew+Cayley done ONCE), as detached leaves with grad.

    Simulates perfect caching of R across microbatches: the timed chain then
    excludes the skew-symmetric build + Cayley entirely.
    """
    Rs = []
    for pl in poet:
        with torch.no_grad():
            R_out, R_in = get_weight_poet_decoupled(
                pl.oft_R_in,
                pl.oft_R_out,
                pl.block_size_in,
                pl.block_size_out,
                pl.rows_in,
                pl.cols_in,
                pl.rows_out,
                pl.cols_out,
            )
        Rs.append((R_in.detach().requires_grad_(True), R_out.detach().requires_grad_(True)))
    return Rs


@torch.compile(fullgraph=True)
def _compiled_chain(
    x, R_in, weight, bias, R_out, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz_in, bsz_out
):
    # mirror of CachedPOETLinear's `_cached_chain_layer_core_decoupled`, but on the
    # fast (non-recompute) chain. MUST be compiled to match the full-POET path
    # (uncompiled, the permute-gather backward is ~10x slower and dominates).
    return chain_layer_x_fast_decoupled(
        x, R_in, weight, bias, R_out, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz_in, bsz_out
    )


def fwd_bwd_chain_only(poet, Rs, in_dims, tokens, dtype, dev):
    """Time only rotation+matmul (R supplied), i.e. POET with skew+Cayley cached.

    Compiled, to be a fair comparison against the compiled full-POET path.
    """
    for pl, (R_in, R_out), i in zip(poet, Rs, in_dims):
        x = torch.randn(tokens, i, device=dev, dtype=dtype, requires_grad=True)
        y = _compiled_chain(
            x,
            R_in,
            pl.weight,
            pl.bias,
            R_out,
            pl.perm_in_inv,
            pl.perm_in,
            pl.perm_out,
            pl.perm_out_inv,
            pl.block_size_in,
            pl.block_size_out,
        )
        y.sum().backward()


def timed_chain_only(poet, Rs, in_dims, tokens, dtype, dev, reps):
    for _ in range(3):
        fwd_bwd_chain_only(poet, Rs, in_dims, tokens, dtype, dev)
    torch.cuda.synchronize()
    s, e = _evt(), _evt()
    ts = []
    for _ in range(reps):
        s.record()
        fwd_bwd_chain_only(poet, Rs, in_dims, tokens, dtype, dev)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def bucket(name: str) -> str:
    n = name.lower()
    # skew-symmetric build: new_zeros + index_put scatter (check BEFORE generic index)
    if "index_put" in n or "new_zero" in n or "_index_put_impl" in n:
        return "skew build"
    if "cayley" in n or "skew" in n or "neumann" in n:
        return "cayley"
    if "bmm" in n or "baddbmm" in n:
        return "bmm (rotation)"
    if (
        "indexing_backward" in n
        or "index" in n
        or "gather" in n
        or "scatter" in n
        or "permute" in n
    ):
        return "gather/permute"
    if (
        name in ("aten::mm", "aten::matmul", "aten::addmm")
        or "nvjet" in n
        or "cutlass" in n
        or "gemm" in n
        or "cublas" in n
    ):
        return "mm (dense)"
    if (
        "elementwise" in n
        or "copy" in n
        or "add" in n
        or "mul" in n
        or "cat" in n
        or "reshape" in n
        or "view" in n
        or "transpose" in n
        or "contiguous" in n
    ):
        return "elementwise/reshape"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=128 * 256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = ap.parse_args()
    assert torch.cuda.is_available(), "need a GPU"
    dev = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]

    print(
        f"GPU={torch.cuda.get_device_name()}  tokens={a.tokens}  block_size={a.block_size}  dtype={a.dtype}  reps={a.reps}"
    )
    poet, dense, in_dims = build(a.block_size, dtype, dev)

    t_dense = timed(dense, in_dims, a.tokens, dtype, dev, a.reps)
    t_poet = timed(poet, in_dims, a.tokens, dtype, dev, a.reps)
    Rs = precompute_R(poet)
    t_chain = timed_chain_only(poet, Rs, in_dims, a.tokens, dtype, dev, a.reps)
    print(f"\n=== end-to-end (one layer's 7 linears, fwd+bwd, median ms) ===")
    print(f"  dense (Adam, trainable W)      : {t_dense:8.3f} ms")
    print(f"  POET  (skew+Cayley every iter) : {t_poet:8.3f} ms   ratio = {t_poet / t_dense:.2f}x")
    print(
        f"  POET  (R cached: chain only)   : {t_chain:8.3f} ms   ratio = {t_chain / t_dense:.2f}x"
    )
    skew_share = (t_poet - t_chain) / max(t_poet - t_dense, 1e-9)
    print(
        f"  --> skew+Cayley = {t_poet - t_chain:.3f} ms = {100*skew_share:.0f}% of the POET overhead;"
        f" rotations+gathers = {t_chain - t_dense:.3f} ms = the rest"
    )

    # op-grouped CUDA-time breakdown for POET
    for _ in range(3):
        fwd_bwd(poet, in_dims, a.tokens, dtype, dev)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fwd_bwd(poet, in_dims, a.tokens, dtype, dev)
        torch.cuda.synchronize()

    groups: dict[str, float] = {}
    total = 0.0
    for ev in prof.key_averages():
        # attribute name varies by torch version (self_cuda_* -> self_device_*)
        cu = (
            float(
                getattr(ev, "self_device_time_total", 0.0)
                or getattr(ev, "self_cuda_time_total", 0.0)
            )
            / 1e3
        )  # us->ms
        if cu <= 0:
            continue
        groups[bucket(ev.key)] = groups.get(bucket(ev.key), 0.0) + cu
        total += cu
    print(f"\n=== POET CUDA self-time by group (over 5 fwd+bwd; total {total:.1f} ms) ===")
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:22s} {v:9.2f} ms   {100 * v / total:5.1f}%")

    print(f"\n=== top kernels (POET, self CUDA time) ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))


if __name__ == "__main__":
    main()
