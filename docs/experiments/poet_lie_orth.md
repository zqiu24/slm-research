# poet_lie_orth — standalone Muon-like orthogonalizing optimizer

Sibling of [`poet_lie_rms`](./poet_lie_rms.md): the standalone `LieOrthMomentum`
optimizer (`q_optimizer=lie_ortho`). Same single-step POET Lie-momentum stack
(`merge_period=1`, `block_count=1`, `reinit_period=-1`, `cayley`, head-aligned), but
the direction→generator transform is **orthogonalization** instead of RMS scaling, per
[docs/muon_orthogonalizing_optimizer_poet.md](../muon_orthogonalizing_optimizer_poet.md).

After the (first-moment) Lie direction `A`, the optimizer orthogonalizes each `b×b`
skew block and scales by `c`, so the rotation planes turn by ~the same angle:

```
X     = orthogonalize(A)          # planes' singular values driven toward 1
oft_R = lr · c · X                # realized per-plane angle ~ lr · lie_ortho_c
```

This discards the gradient's *relative* per-plane magnitudes (keeps only the
subspace) — Muon's bet, applied to rotational updates. First-moment-only by default
(a second moment is partially undone by orthogonalization, docs §4).

`lie_ortho_method`:
- **`muon`** (default) — Muon's quintic Newton–Schulz then a `½(X−Xᵀ)` cleanup. NS
  preserves skew on a skew input; it democratizes the spectrum into a **band** around
  1 (cond ≈ 1.5) in ~5 steps. Cheap; `c` is a *nominal* angle (band ≈ 0.7–1.1× target).
- **`spectral`** — exact `A(−A²)^{-1/2}`; drives every singular value to 1 so `c` is
  exactly the angle. Needs `lie_ortho_ns_steps ≈ 20` (≈4× the cost).

Run head-to-head vs `poet_lie_rms` to test whether the gradient's relative per-plane
angles are signal or noise for rotational updates (docs §7) — and `muon` vs `spectral`
to test whether a cheap band is as good as exact equalization.

## Single-step fast path (`single_step_fast`)

With `merge_period=1` the rotation is folded into `W` and `oft_R` zeroed every
step, so `R=Identity` at every forward. `optim.poet.single_step_fast=true`
collapses the identity-rotation chain to a permuted GEMM and computes the `oft_R`
gradient in closed form (factor-2 Cayley Jacobian), removing ~3x the base-GEMM
rotation FLOPs from the MLP (and non-head-aligned attention). Mathematically
identical training. A/B:

```bash
# baseline (chain)
codexlog lieorth_chain bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true
# fast path
codexlog lieorth_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_fast=true
```

Compare steady-state `elapsed time per iteration (ms)` (skip first ~20 iters for
compile warmup) and confirm the loss curves overlap.
