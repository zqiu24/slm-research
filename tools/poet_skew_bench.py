# ruff: noqa
"""Microbench: current scatter-based skew build vs a gather-based reformulation.

Both compiled, fwd+bwd, at the 300m model's real oft_R block shapes. Reports
median ms and speedup. The gather version is bit-exact (verified separately) —
this only measures whether inductor's gather codegen beats its scatter codegen.

Run (single GPU, training venv):
    CUDA_VISIBLE_DEVICES=0 python tools/poet_skew_bench.py
"""

from __future__ import annotations

import torch

# (r blocks, block_size) pairs that occur in the 300m layer at block_size=256:
#   q/o: in(4) out(4); k/v: in(4) out(1); gate/up: in(4) out(10); down: in(10) out(4)
SHAPES = [(1, 256), (4, 256), (10, 256)]
REPS = 50


@torch.compile(fullgraph=True)
def skew_scatter(vec, b, rows, cols):
    m = vec.new_zeros(vec.shape[0], b, b)
    m[:, rows, cols] = vec
    return m - m.transpose(-2, -1)


@torch.compile(fullgraph=True)
def skew_gather(vec, b, gidx, sign):
    return (vec[:, gidx] * sign).view(vec.shape[0], b, b)


def make_map(b, rows, cols, device, dtype):
    n = rows.numel()
    bb = b * b
    gidx = torch.zeros(bb, dtype=torch.long, device=device)
    sign = torch.zeros(bb, dtype=dtype, device=device)
    ar = torch.arange(n, device=device)
    up = rows.long() * b + cols.long()
    lo = cols.long() * b + rows.long()
    gidx[up] = ar
    sign[up] = 1.0
    gidx[lo] = ar
    sign[lo] = -1.0
    return gidx, sign


def timed(fn, *args):
    for _ in range(5):
        v = args[0]
        v.grad = None
        y = fn(*args)
        y.sum().backward()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(REPS):
        args[0].grad = None
        s.record()
        y = fn(*args)
        y.sum().backward()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    assert torch.cuda.is_available()
    dev, dtype = "cuda", torch.bfloat16
    print(f"GPU={torch.cuda.get_device_name()}  dtype=bf16  reps={REPS}\n")
    print(f"{'r x b':>10}  {'scatter ms':>11}  {'gather ms':>11}  {'speedup':>8}")
    for r, b in SHAPES:
        n = b * (b - 1) // 2
        rows, cols = torch.triu_indices(b, b, 1, device=dev)
        rows, cols = rows.to(torch.int32), cols.to(torch.int32)
        gidx, sign = make_map(b, rows, cols, dev, dtype)
        v1 = torch.randn(r, n, device=dev, dtype=dtype, requires_grad=True)
        v2 = torch.randn(r, n, device=dev, dtype=dtype, requires_grad=True)
        t_s = timed(skew_scatter, v1, b, rows, cols)
        t_g = timed(skew_gather, v2, b, gidx, sign)
        print(f"{r:4d} x {b:<4d}  {t_s:11.4f}  {t_g:11.4f}  {t_s / t_g:7.2f}x")


if __name__ == "__main__":
    main()
