# POET-X × Pion: Single-Step Implementation Pipeline

A design spec for the next iteration of POET-X. The goal: keep POET's
parameterization (the source of **memory efficiency**) while importing Pion's
update geometry (the source of **performance / freshness**), and add a
selectively-sharded merge for **throughput** under DDP.

The three pillars and where each comes from:

- **Memory efficiency** — the block-diagonal skew-symmetric parameterization,
  upper-triangular storage, input-centric formulation, checkpointing. Unchanged.
- **Performance** — interval-1 (merge every step) + Lie-algebra momentum +
  RMS-normalized step size, all from Pion. New.
- **Throughput** — block-sharded merge across DDP replicas. New.

---

## 0. Notation

- `W ∈ R^{dout×din}` — a live weight matrix (no frozen `W0` anchor anymore).
- `G = ∇_W f(W)` — the weight gradient from autograd (already all-reduced under DDP).
- Block size `b`; left side has `⌈dout/b⌉` blocks, right side `⌈din/b⌉` blocks.
- `Q` — per-block skew-symmetric matrix, stored as upper-triangular only
  (`b(b-1)/2` params per block). This is the *only* thing carrying the rotation.
- `Ψ` — block-stochastic permutation (resampled **every step**), defining which
  neurons fall into which block this step.
- `Cay_k(Q)` — Cayley–Neumann parameterization of order `k`. At interval 1 the
  per-step angle is tiny, so `k` can drop to 1 (`G ≈ I + 2Q`) or 2.

---

## 1. The core conceptual shift

Original POET held `W0` fixed and accumulated `Q` over a long merge interval
(reset gap = 400), so `Q` grew large → needed CNP `k=3`, and the optimizer ran
on `Q` in ambient space against a stale anchor.

**New scheme: merge interval = 1.** Each step `Q` is born at 0 (identity),
takes one small step, is exponentiated, merged into `W`, and reset. This makes
the per-step rotation small, which is exactly Pion's "start from identity each
step so errors don't compound" condition — and it unlocks:

- low-order Cayley/exp (k=1 or 2 instead of 3),
- Lie-algebra momentum with **no parallel transport** (buffer always at identity),
- alternating single-sided updates for free (half the optimizer state/step).

Over-parameterization is retained **only in the memory sense** (compact
block-skew parameterization), not in the "persistent auxiliary factors reshape
the landscape" sense. That justification is dropped deliberately.

---

## 2. The tangent (Lie-algebra) gradient

Instead of differentiating `f(G(Q))` through the CNP chain rule (which injects
the CNP Jacobian's curvature and lands you in ambient coordinates), form the
tangent gradient directly from the current weight and its gradient, then project
to skew-symmetric. This is Pion's `G^in` / `G^out`:

```
G_out = G W^T − (G W^T)^T = G W^T − W G^T     # left-factor tangent  (dout×dout)
G_in  = W^T G − (W^T G)^T = W^T G − G^T W     # right-factor tangent (din×din)
```

Both are skew-symmetric by construction → they live in the Lie algebra so(n).
In the block-diagonal setting you only ever need the **block-diagonal entries**
of these: for block `j` owning neuron index set `S_j` (from this step's Ψ),
compute `G_out[S_j, S_j]` using the corresponding rows of `W` and `G`. Never
materialize the full dout×dout matrix.

---

## 3. Lie-algebra momentum (the Pion import you asked about)

Because every step starts at the identity, the momentum buffer is always
expressed in the *same* tangent space (so(n) at I). So plain momentum on the
skew gradient stays in the algebra — no transport needed. Per block, per active
side:

```
# first moment, on the skew-symmetric tangent gradient
M ← β1 · M + (1 − β1) · G_skew          # M skew-symmetric, stored upper-tri

# second moment: SCALAR per block (NOT element-wise), to preserve isotropy
v ← β2 · v + (1 − β2) · ||G_skew||_F^2   # one scalar per block

# normalized tangent direction (faithful skew direction, not coord-warped)
A ← − M / ( sqrt(v) + ε )
```

Why scalar `v` and not AdamW's element-wise `v`: the entries of a
skew-symmetric matrix jointly parameterize one rotation; the natural so(n) metric
is the isotropic Frobenius inner product. Per-entry rescaling distorts the
rotation *direction*. A scalar (per-block) second moment keeps adaptivity without
warping geometry. This is the single change Pion's ablation isolates as a
consistent win — implement it first, in isolation, to read its effect cleanly.

(Optional, closer to Pion's exact algorithm: keep element-wise `v` but then
re-normalize the *whole* block update by its Frobenius RMS via the α step below.
Ablate scalar-v vs elementwise-v+α — start with scalar-v.)

---

## 4. RMS step-size normalization (replaces global γ = 0.5)

Pion's per-matrix scale-proportional step. Replaces the single global γ with a
normalization that makes the rotational magnitude comparable across matrices of
different dims, which is what unlocks larger learning rates:

```
α ← c · sqrt(dout · din) / ( || A · W ||_F + ε )     # two-sided
# single-sided (alternating), active = out:
α ← c · sqrt(dout · din) / ( || A_out · W ||_F + ε )
```

`c` is the one new scalar hyperparameter (RMS target). Tune once; with the μP
spectral condition it should transfer across widths.

---

## 5. Small-angle map: low-order Cayley / exp

At interval 1 the angle `η·α·A` is small, so:

```
# Option A — second-order exp (Pion):   exp(X) ≈ I + X + ½ X²
# Option B — Cayley order-1 (CNP k=1):  Cay(X) ≈ I + 2X     (your StiefelAdam-era form)
# Option C — Cayley order-2 (CNP k=2):  better orthogonality, slightly more cost
```

Trade-off: exp/CNP-low are only *approximately* orthogonal; true Cayley
`(I−X/2)^{-1}(I+X/2)` is *exactly* orthogonal (exact spectrum preservation) but
needs the inverse. At small angles all agree numerically — pick by whether you
want an exactness *guarantee* for the stability story. Make `k` a knob; default
to order-2.

---

## 6. Alternating single-sided update (efficiency lever)

Update **one** side per step: out-side on even steps, in-side on odd. Pion: ~0.23%
loss cost, ~half optimizer compute. In our setting it also halves the per-step
optimizer **state** (one side's M, v only).

Caveat to measure: alternation + block-stochastic resampling are two stochastic
sources; coverage of all neuron pairs (effective full-matrix expressivity) is
slower than two-sided-every-step. Pion had no block structure, so their 0.23% is
a *lower bound* for us. Ablate two-sided vs alternating on the loss-curve shape,
not just final PPL.

---

## 7. Per-step pipeline (single weight matrix)

```
INPUT: W_t, side ∈ {out,in} (alternates), momentum state (M,v) for active side
                                            , RNG seed s_t

1.  Ψ_t          ← sample_block_permutation(s_t)          # per-step block-stochastic
2.  G_t          ← autograd weight gradient (all-reduced under DDP)
3.  for each block j owned (see §8 for sharding):
       S_j       ← Ψ_t block index set
       G_skew_j  ← skew( active-side tangent grad on S_j )   # §2
       M_j       ← β1 M_j + (1−β1) G_skew_j                   # §3
       v_j       ← β2 v_j + (1−β2) ||G_skew_j||_F^2
       A_j       ← − M_j / (sqrt(v_j)+ε)
4.  α_t          ← c·sqrt(dout·din) / (|| A·W_t ||_F + ε)     # §4 (one reduction)
5.  for each block j owned:
       Gblk_j    ← Cay_k( η α_t A_j )                          # §5, k=1 or 2
6.  W_{t+1}      ← apply block rotations to active side of W_t # block-local matmul
                   (use permutation-merge trick: pre-permute W, Eq.6)
7.  reset Q (implicit: A_j discarded; M,v persist for that side)
8.  swap side for t+1
```

Everything in steps 3–6 runs on your existing batch-parallel block kernels and
permutation kernels. The transplant is: replace "accumulate Q over 400 steps +
CNP k=3" with steps 2–7 above.

---

## 8. Block-sharded merge under DDP (throughput lever)

Status quo (DDP): after gradient all-reduce, every GPU redundantly computes the
*identical* merge N times. The block-diagonal structure makes the merge
embarrassingly parallel along the block dimension (no cross-block dependency), so
shard it.

Two schemes:

- **Scheme A — shard compute, all-gather W.** GPU `k` owns block-set `k`,
  computes only those `Gblk`, applies to its row/col slice of `W`, then
  `all_gather` the updated `W` slices so every replica has full `W` for the next
  forward.
  - Net win **iff** merge_compute > all_gather_cost. Merge compute scales
    ~O(dout·b²); all-gather ~O(dout·din). So this pays off at **large b (512)** —
    which conveniently is also the better-PPL regime.

- **Scheme B — replicate (status quo).** No extra comm.

Decision rule: **profile the merge as a fraction of step wall-clock first.**
POET-X already made the merge cheap (1.38ms vs POET's 10.59ms). If the merge is
<5% of step time, Scheme A's all-gather will cost more than it saves — skip it.

Best version — fuse with alternating (§6): single-sided merge is block-row-local,
so GPU `k` owns row-block-set `k`, builds only those blocks, applies only to its
rows. Optimizer state (M,v for the active side) is then **sharded by block across
replicas** → already-halved-by-alternating state, further divided by N. Helps the
multi-GPU memory/throughput story (Table 9/10), not single-GPU numbers.

Framing for the paper: *selective sharding of only the orthogonal-merge step,
preserving DDP's communication profile everywhere else.* More defensible than
FSDP because you shard the one op that is block-parallel and adds no cross-device
dependency, rather than sharding every layer and eating collective overhead.

---

## 9. Ablation order (each isolates one cause)

1. **Lie-algebra momentum alone** — keep interval 400, CNP k=3, two-sided; only
   change §2+§3 (tangent grad + scalar-v momentum). Cleanest read on the headline
   Pion import. Target: measurable move from 12.05 toward Muon's 11.45.
2. **Interval 1 + low-order Cayley** — add §1+§5; confirm small-angle CNP-k1/2
   matches k=3 quality at lower cost.
3. **RMS α vs global γ** — swap §4 in; check larger-LR stability.
4. **Alternating vs two-sided** — §6; measure loss-curve shape + final PPL, watch
   for block-stochastic coverage slowdown.
5. **Sharded merge** — §8 Scheme A at b=512; metric is end-to-end tokens/s, not
   merge-time-in-isolation.

Calibration target (Pion's headline): LLaMA-1.3B / 54B C4 tokens, Pion val loss
2.7350 ≈ Muon 2.7225, AdamW 2.7700; Pion best avg downstream (47.69 vs Muon
46.34). POET should aim to land at Muon on loss, ideally above it downstream.

---

## 10. Open questions to resolve with experiments

- **Tangent grad against current W vs frozen anchor.** §2 uses current `W`
  (couples the two factors, more correct, more compute). A cheaper variant uses a
  short-lived anchor. At interval 1 they nearly coincide — verify.
- **Does alternation stack cleanly with block-stochastic resampling?** (§6)
- **Is the merge actually a bottleneck worth sharding?** (§8 — profile first.)
- **Lowest viable CNP order at interval 1** — is k=1 (`I+2Q`) enough, or is k=2
  needed for stable orthogonality over long runs?
