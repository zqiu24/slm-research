"""Decide whether POET's huge ||dL/d oft_R|| is a BACKWARD BUG or inherent.

Finite-difference gradcheck on the *exact* training ops (cayley + chain),
run in EAGER fp32 (no torch.compile) so the numeric slope is clean.

Finite-difference only uses the FORWARD, so it is an independent ground
truth for the gradient. Compare it to autograd's analytic gradient:

    ratio = analytic / numeric
      ~1     -> backward is CORRECT; the ~1e5 norm is REAL (inherent
                steepness of large-block Cayley) -> fix via block size / LR.
      ~1e4   -> the hand-written VJP is over-scaled -> BACKWARD BUG in the
                poet kernels (cayley_backward / chain_..._decoupled backward).

Run on a free GPU (uses the same poet_torch the training imports):
    CUDA_VISIBLE_DEVICES=0 python tools/poet_gradcheck.py
"""

import torch
from poet_torch import POETLinear
from poet_torch.poet_layer import (
    chain_layer_x_checkpoint_mem_o2_decoupled,
    get_weight_poet_decoupled,
)

torch.manual_seed(0)
DEV = "cuda"
DT = torch.float32


def build(in_f, out_f, block_count):
    """One POETLinear with the real init_type='normalized' (row-norm weight,
    oft_R = 0 => R = I, exactly the step-1 state)."""
    pl = POETLinear(
        in_features=in_f,
        out_features=out_f,
        block_count=block_count,
        bias=False,
        device=DEV,
        dtype=DT,
    )
    with torch.no_grad():
        w = torch.randn(out_f, in_f, device=DEV, dtype=DT) * 0.02
        w = w / torch.norm(w, dim=1, keepdim=True)
        pl.weight.copy_(w)
        pl.oft_R_in.zero_()
        pl.oft_R_out.zero_()
    return pl


def forward_eager(pl, x):
    """Identical math to POETLinear.forward but WITHOUT @torch.compile, so the
    finite-difference re-evaluations are not perturbed by graph caching."""
    R_out, R_in = get_weight_poet_decoupled(  # noqa: N806
        pl.oft_R_in,
        pl.oft_R_out,
        pl.block_size_in,
        pl.block_size_out,
        pl.rows_in,
        pl.cols_in,
        pl.rows_out,
        pl.cols_out,
    )
    return chain_layer_x_checkpoint_mem_o2_decoupled(
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


def loss_of(pl, x, g):
    # fixed g => dL/dy = g exactly; sum reduction. Cast to fp64 for the scalar.
    y = forward_eager(pl, x)
    return (y.double() * g.double()).sum()


def check(in_f, out_f, block_count, n_probe=12, eps=1e-3):
    pl = build(in_f, out_f, block_count)
    B, S = 4, 4  # noqa: N806
    x = torch.randn(B, S, in_f, device=DEV, dtype=DT)
    g = torch.randn(B, S, out_f, device=DEV, dtype=DT)

    pl.zero_grad(set_to_none=True)
    L = loss_of(pl, x, g)  # noqa: N806
    L.backward()
    g_in = pl.oft_R_in.grad.detach().clone()
    g_out = pl.oft_R_out.grad.detach().clone()

    print(
        f"\n### in={in_f} out={out_f} block_count={block_count} "
        f"bs_in={pl.block_size_in} bs_out={pl.block_size_out}"
    )
    print(
        f"  ||grad oft_R_in||  = {g_in.norm():.4e}  "
        f"rms/elem={g_in.pow(2).mean().sqrt():.4e}  numel={g_in.numel()}"
    )
    print(
        f"  ||grad oft_R_out|| = {g_out.norm():.4e}  "
        f"rms/elem={g_out.pow(2).mean().sqrt():.4e}  numel={g_out.numel()}"
    )

    # Finite-difference the big-block side (oft_R_out) on random entries.
    P = pl.oft_R_out  # noqa: N806
    nrow, ncol = P.shape
    mism = 0
    print(f"  finite-diff vs analytic on oft_R_out (eps={eps}):")
    for _ in range(n_probe):
        r = int(torch.randint(0, nrow, (1,)))
        c = int(torch.randint(0, ncol, (1,)))
        with torch.no_grad():
            P[r, c] += eps
        Lp = loss_of(pl, x, g).item()  # noqa: N806
        with torch.no_grad():
            P[r, c] -= 2 * eps
        Lm = loss_of(pl, x, g).item()  # noqa: N806
        with torch.no_grad():
            P[r, c] += eps  # restore
        num = (Lp - Lm) / (2 * eps)
        ana = g_out[r, c].item()
        ratio = ana / num if abs(num) > 1e-7 else float("inf")
        ok = 0.5 < abs(ratio) < 2.0 if abs(num) > 1e-7 else True
        mism += 0 if ok else 1
        tag = "" if ok else "   <-- MISMATCH"
        print(
            f"    [{r:>3},{c:>6}] analytic={ana:+.4e}  numeric={num:+.4e}  "
            f"ratio={ratio:+.4f}{tag}"
        )
    verdict = (
        "backward CORRECT -> magnitude is INHERENT (block size is the lever)"
        if mism == 0
        else "backward MISMATCH -> hand-written VJP scale BUG in poet kernels"
    )
    print(f"  >>> {mism}/{n_probe} mismatched.  {verdict}")


if __name__ == "__main__":
    print("POET gradcheck — analytic (autograd) vs finite-difference, fp32 eager")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    # Same hidden=1024 / ffn=2560 as the 300m run. Sweep block_count to show
    # how the gradient magnitude scales with block size.
    for shape in [
        (1024, 2560, 4),  # ffn-like, bs_out=640  (the worst case in the run)
        (1024, 1024, 4),  # attn-like, bs=256     (the run's setting)
        (1024, 1024, 8),  # bs=128
        (1024, 1024, 16),  # bs=64
        (1024, 1024, 32),  # bs=32
    ]:
        try:
            check(*shape)
        except Exception:
            import traceback

            print(f"  shape {shape} FAILED:")
            traceback.print_exc()
