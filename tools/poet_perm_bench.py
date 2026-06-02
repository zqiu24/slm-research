# ruff: noqa
"""Is the permutation BACKWARD (scatter-add) the bottleneck, and does a
gather-backward fix it?

A permutation gather y = x[..., perm] has, in autograd, a scatter-add backward
(`indexing_backward_kernel`) because indexing assumes possible duplicate indices.
A permutation has none, so its true backward is the inverse-perm gather. This
times both, compiled, fwd+bwd, at the real activation size.

Run:  CUDA_VISIBLE_DEVICES=0 python tools/poet_perm_bench.py
"""

from __future__ import annotations

import torch

REPS = 80


class PermGatherBwd(torch.autograd.Function):
    """y = x[..., perm]; backward is the inverse-perm GATHER (not scatter-add)."""

    @staticmethod
    def forward(ctx, x, perm, inv_perm):
        ctx.save_for_backward(inv_perm)
        return x.index_select(-1, perm)

    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(-1, inv), None, None


def timed(fn, x, reps=REPS, warmup=10):
    for _ in range(warmup):
        x.grad = None
        fn().sum().backward()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        x.grad = None
        s.record()
        fn().sum().backward()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    assert torch.cuda.is_available()
    dev, dtype = "cuda", torch.bfloat16
    N = 128 * 256
    print(f"GPU={torch.cuda.get_device_name()}  tokens={N}  dtype=bf16  reps={REPS}")
    print(f"  (2 perms per call: input + output, like the chain)\n")
    print(f"  {'dim':>6}  {'plain idx (scatter bwd)':>24}  {'gather bwd':>12}  {'speedup':>8}")

    for dim in (1024, 2560):
        x = torch.randn(N, dim, device=dev, dtype=dtype, requires_grad=True)
        perm = torch.randperm(dim, device=dev)
        inv = torch.argsort(perm)

        @torch.compile(fullgraph=True)
        def plain(x):  # two perms, autograd scatter backward
            return (x[..., perm])[..., perm]

        @torch.compile(fullgraph=True)
        def gbwd(x):  # two perms, custom gather backward
            return PermGatherBwd.apply(PermGatherBwd.apply(x, perm, inv), perm, inv)

        try:
            t_plain = timed(lambda: plain(x), x)
        except Exception as ex:  # noqa: BLE001
            t_plain = float("nan")
            print(f"  plain compile failed: {ex}")
        try:
            t_g = timed(lambda: gbwd(x), x)
        except Exception as ex:  # noqa: BLE001
            t_g = float("nan")
            print(f"  gather-bwd compile failed: {ex}")
        sp = t_plain / t_g if t_g == t_g and t_g > 0 else float("nan")
        print(f"  {dim:>6}  {t_plain:>24.4f}  {t_g:>12.4f}  {sp:>7.2f}x")


if __name__ == "__main__":
    main()
