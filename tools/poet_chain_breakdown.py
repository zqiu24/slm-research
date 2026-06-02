# ruff: noqa
"""Where does POET's fwd+bwd time go? Cumulative build-up, each stage compiled.

Builds the fast-chain forward one operation at a time and times fwd+bwd at each
stage. Each stage's *delta* over the previous is the marginal cost of adding that
op; the stages sum to the full POET layer. Differential timing (CUDA events,
compiled per stage) — not profiler self-time attribution.

Stages (all fwd+bwd, frozen W like real POET; R_in/R_out are requires_grad
leaves so the rotation backward computes grad_R, matching real POET):
  0 matmul            y = x @ W^T                         (frozen W -> grad_x only)
  1 + rotate-in       y = rot_in(x) @ W^T
  2 + rotate-out      y = rot_out(rot_in(x) @ W^T)
  3 + gathers         + perm_in_inv / perm_out            (= full fast chain, R cached)
  4 + skew+Cayley     R built from oft_R each call        (= full POET, uncached)
  ref: Adam matmul    trainable W (grad_x + grad_W)       (dense baseline)

Run:  CUDA_VISIBLE_DEVICES=0 python tools/poet_chain_breakdown.py
"""

from __future__ import annotations

import sys

import torch

_REPO = __file__.rsplit("/tools/", 1)[0]
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from poet_torch.poet_layer import get_weight_poet_decoupled, pytorch_skew_symmetric  # noqa: E402

# (name, in_features, out_features) — the two distinct 300m POET layer shapes.
SHAPES = [("attn q/o  1024x1024", 1024, 1024), ("ffn gate/up 1024x2560", 1024, 2560)]
BATCH = (128, 256)  # 32768 tokens, matches micro_batch=128 seq=256
BLOCK = 256
REPS = 60


def rot(x, R, b):
    """Block-diagonal rotation: x[..., dim] -> reshape [N,r,b], bmm with R[r,b,b]."""
    N = x.shape[0]
    r = R.shape[0]
    return torch.bmm(x.reshape(N, r, b).transpose(0, 1), R).transpose(0, 1).reshape(N, r * b)


def timed(fn, *leaves, reps=REPS, warmup=8):
    for _ in range(warmup):
        for L in leaves:
            L.grad = None
        fn().sum().backward()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        for L in leaves:
            L.grad = None
        s.record()
        fn().sum().backward()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def run_shape(name, in_f, out_f, dev, dtype):
    b = BLOCK
    r_in, r_out = in_f // b, out_f // b
    n_el_in, n_el_out = b * (b - 1) // 2, b * (b - 1) // 2
    N = BATCH[0] * BATCH[1]

    W = torch.randn(out_f, in_f, device=dev, dtype=dtype)  # frozen base
    Wt = torch.randn(out_f, in_f, device=dev, dtype=dtype, requires_grad=True)  # Adam ref
    perm_in_inv = torch.randperm(in_f, device=dev)
    perm_out = torch.randperm(out_f, device=dev)
    rows_in, cols_in = torch.triu_indices(b, b, 1, device=dev)
    rows_out, cols_out = torch.triu_indices(b, b, 1, device=dev)

    def fresh_x():
        return torch.randn(N, in_f, device=dev, dtype=dtype, requires_grad=True)

    def fresh_R():
        R_in = torch.randn(r_in, b, b, device=dev, dtype=dtype, requires_grad=True)
        R_out = torch.randn(r_out, b, b, device=dev, dtype=dtype, requires_grad=True)
        return R_in, R_out

    oft_in = torch.randn(r_in, n_el_in, device=dev, dtype=dtype, requires_grad=True) * 0.01
    oft_out = torch.randn(r_out, n_el_out, device=dev, dtype=dtype, requires_grad=True) * 0.01
    oft_in = oft_in.detach().requires_grad_(True)
    oft_out = oft_out.detach().requires_grad_(True)

    x = fresh_x()
    R_in, R_out = fresh_R()

    @torch.compile(fullgraph=True)
    def s_ref(x):  # Adam: trainable W
        return x @ Wt.t()

    @torch.compile(fullgraph=True)
    def s0(x):  # matmul, frozen W
        return x @ W.t()

    @torch.compile(fullgraph=True)
    def s1(x, R_in):  # + rotate-in
        return rot(x, R_in, b) @ W.t()

    @torch.compile(fullgraph=True)
    def s2(x, R_in, R_out):  # + rotate-out
        return rot(rot(x, R_in, b) @ W.t(), R_out, b)

    @torch.compile(fullgraph=True)
    def s3(x, R_in, R_out):  # + gathers (= full fast chain)
        xp = x[..., perm_in_inv]
        y = rot(rot(xp, R_in, b) @ W.t(), R_out, b)
        return y[..., perm_out]

    @torch.compile(fullgraph=True)
    def s4(x, oft_in, oft_out):  # + skew+Cayley (= full POET)
        R_out_, R_in_ = get_weight_poet_decoupled(
            oft_in, oft_out, b, b, rows_in, cols_in, rows_out, cols_out
        )
        xp = x[..., perm_in_inv]
        y = rot(rot(xp, R_in_, b) @ W.t(), R_out_, b)
        return y[..., perm_out]

    t_ref = timed(lambda: s_ref(x), x, Wt)
    t0 = timed(lambda: s0(x), x)
    t1 = timed(lambda: s1(x, R_in), x, R_in)
    t2 = timed(lambda: s2(x, R_in, R_out), x, R_in, R_out)
    t3 = timed(lambda: s3(x, R_in, R_out), x, R_in, R_out)
    t4 = timed(lambda: s4(x, oft_in, oft_out), x, oft_in, oft_out)

    print(f"\n### {name}  (tokens={N}, block={b}, r_in={r_in}, r_out={r_out}) ###")
    print(f"  {'stage':<26}{'fwd+bwd ms':>12}{'Δ (this op)':>14}{'% of full':>11}")
    rows = [
        ("0 matmul (frozen)", t0, t0),
        ("1 + rotate-in", t1, t1 - t0),
        ("2 + rotate-out", t2, t2 - t1),
        ("3 + gathers (=chain)", t3, t3 - t2),
        ("4 + skew+Cayley (=POET)", t4, t4 - t3),
    ]
    for label, cum, delta in rows:
        print(f"  {label:<26}{cum:>12.3f}{delta:>14.3f}{100*delta/t4:>10.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'POET full':<26}{t4:>12.3f}")
    print(f"  {'Adam ref (trainable W)':<26}{t_ref:>12.3f}   -> POET/Adam = {t4/t_ref:.2f}x")


def main():
    assert torch.cuda.is_available()
    dev, dtype = "cuda", torch.bfloat16
    print(f"GPU={torch.cuda.get_device_name()}  dtype=bf16  reps={REPS}  batch={BATCH}")
    for name, i, o in SHAPES:
        run_shape(name, i, o, dev, dtype)


if __name__ == "__main__":
    main()
