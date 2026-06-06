"""Microbenchmark for LieOrthMomentum.step on representative 60m shapes.

Isolates: (a) full optimizer step, (b) skew<->vec conversions only, (c) NS only.
Run on a GPU node (uses the training env). CPU run works too but is not
representative of the H100 timing.

    PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
    $PY tools/lie_orth_profile.py --device cuda --steps 30
"""

from __future__ import annotations

import argparse
import time

import torch

from src.diag.skew_conditioning import skew_to_vec, vec_to_skew
from src.optim.poet_lie_orth import LieOrthMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction

# 60m, block_count=1: per layer ~3 FFN R_out blocks of 1536 and ~9 blocks of 512.
LAYERS = 18
BLOCK_SIZES = ([1536] * 3 + [512] * 9) * LAYERS


def _make_params(device):
    ps = []
    for b in BLOCK_SIZES:
        ne = b * (b - 1) // 2
        p = torch.nn.Parameter(torch.zeros(1, ne, device=device))
        ps.append(p)
    return ps


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    dev = args.device

    ps = _make_params(dev)
    grads = [torch.randn_like(p) for p in ps]
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=3e-3)],
        ortho_c=8.0,
        ortho_method="muon",
        ortho_ns_steps=5,
    )

    # (a) full step
    for p, g in zip(ps, grads, strict=False):
        p.grad = g
    opt.step()  # warmup (allocs buffers)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for p, g in zip(ps, grads, strict=False):
            p.grad = g
        opt.step()
    _sync(dev)
    full = (time.perf_counter() - t0) / args.steps

    # (b) conversions only: vec_to_skew -> skew_to_vec round-trip
    dirs = [torch.randn(1, b * (b - 1) // 2, device=dev) for b in BLOCK_SIZES]
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for d, b in zip(dirs, BLOCK_SIZES, strict=False):
            skew_to_vec(vec_to_skew(d, b), b)
    _sync(dev)
    conv = (time.perf_counter() - t0) / args.steps

    # (c) NS only (on pre-built dense skew)
    skews = [vec_to_skew(d, b) for d, b in zip(dirs, BLOCK_SIZES, strict=False)]
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for s in skews:
            orthogonalize_skew_direction(s, method="muon", ns_steps=5)
    _sync(dev)
    ns = (time.perf_counter() - t0) / args.steps

    print(f"device={dev} steps={args.steps}")
    print(f"  full step          : {full * 1000:8.1f} ms")
    print(f"  conversions only   : {conv * 1000:8.1f} ms")
    print(f"  NS only            : {ns * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
