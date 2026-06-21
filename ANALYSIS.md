# POET Two-Sided Coordination: Diagnostic and Experiment Plan

## Objective

Determine how to coordinate the necessary input-side and output-side POET
rotations more effectively than naive simultaneous updates.

The primary question is:

> Why does alternating the written side outperform updating both sides
> simultaneously, even when total weight-space movement is matched?

The experiments should distinguish among:

1. stale second-side directions;
2. destructive finite-step interaction;
3. overlap or cancellation between the two update directions;
4. suboptimal relative angle allocation;
5. momentum filtering effects;
6. network-wide synchronization effects.

---

## Established observations

The following results are already known:

- Input-only POET performs much worse.
- Output-only POET performs much worse.
- Both sides are therefore necessary.
- Updating both sides simultaneously is worse than alternating writes.
- Alternating remains better after matching approximate per-step
  weight-space movement.
- Alternating only works when both input- and output-side momenta continue
  receiving fresh gradients.
- Freezing the inactive side's momentum causes a major regression.

This suggests that the two sides are:

- complementary in expressivity;
- coupled through the current weight;
- sensitive to when their directions are evaluated;
- sensitive to temporal momentum coherence.

---

# 1. Mathematical setup

For a weight matrix

\[
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}},
\]

the local two-sided POET update is

\[
\Delta W
\approx
A_{\mathrm{out}}W + WA_{\mathrm{in}},
\]

where

\[
A_{\mathrm{out}}^\top=-A_{\mathrm{out}},
\qquad
A_{\mathrm{in}}^\top=-A_{\mathrm{in}}.
\]

The corresponding generator-space gradient signals are

\[
H_{\mathrm{out}} = GW^\top,
\]

\[
H_{\mathrm{in}} = W^\top G,
\]

where

\[
G = \frac{\partial L}{\partial W}.
\]

The block-usable raw signals are

\[
K_{\mathrm{out}}(W)
=
\operatorname{block}
\left(
\operatorname{skew}(GW^\top)
\right),
\]

\[
K_{\mathrm{in}}(W)
=
\operatorname{block}
\left(
\operatorname{skew}(W^\top G)
\right).
\]

Here,

\[
\operatorname{skew}(H)
=
\frac{H-H^\top}{2}.
\]

---

# 2. Primary diagnostic: stale versus fresh second-side direction

## Hypothesis

When both sides are updated simultaneously, both directions are computed from
the same old weight \(W\).

After the first side changes \(W\), the correct direction for the second side
may be materially different.

Alternating may work because it approximates a Gauss–Seidel update:

1. update one side;
2. fold it into \(W\);
3. recompute gradients;
4. update the other side from the changed weight.

---

## 2.1 Output-first probe

At a diagnostic checkpoint:

1. Freeze:
   - model parameters;
   - optimizer state;
   - minibatch;
   - dropout/random-number state.

2. Run forward and backward at the original weight \(W\).

3. Compute:

   \[
   K_{\mathrm{out}}^{\mathrm{before}}
   =
   K_{\mathrm{out}}(W),
   \]

   \[
   K_{\mathrm{in}}^{\mathrm{before}}
   =
   K_{\mathrm{in}}(W).
   \]

4. Construct the actual output-side optimizer direction:

   \[
   A_{\mathrm{out}}
   =
   \mathcal O_{\mathrm{out}}
   \left(
   m_{\mathrm{out}},
   K_{\mathrm{out}}^{\mathrm{before}}
   \right),
   \]

   where \(\mathcal O\) includes:

   - momentum;
   - optional Nesterov look-ahead;
   - Newton–Schulz orthogonalization;
   - angle rescaling.

5. Form a virtual output-updated weight:

   \[
   W_o
   =
   R_{\mathrm{out}}W.
   \]

6. Replay the same minibatch at \(W_o\), with identical random state.

7. Recompute the input-side signal:

   \[
   K_{\mathrm{in}}^{\mathrm{after}}
   =
   K_{\mathrm{in}}(W_o).
   \]

8. Construct stale and fresh candidate input momentum states from the same
   pre-update momentum \(m_{\mathrm{in}}^{-}\):

   \[
   m_{\mathrm{in}}^{\mathrm{stale}}
   =
   \beta m_{\mathrm{in}}^{-}
   +
   (1-\beta)K_{\mathrm{in}}^{\mathrm{before}},
   \]

   \[
   m_{\mathrm{in}}^{\mathrm{fresh}}
   =
   \beta m_{\mathrm{in}}^{-}
   +
   (1-\beta)K_{\mathrm{in}}^{\mathrm{after}}.
   \]

9. Construct the final stale and fresh `lie_ortho` directions:

   \[
   A_{\mathrm{in}}^{\mathrm{stale}}
   =
   \mathcal O
   \left(
   m_{\mathrm{in}}^{\mathrm{stale}}
   \right),
   \]

   \[
   A_{\mathrm{in}}^{\mathrm{fresh}}
   =
   \mathcal O
   \left(
   m_{\mathrm{in}}^{\mathrm{fresh}}
   \right).
   \]

---

## 2.2 Input-first probe

Repeat the same procedure in reverse:

1. compute both signals at \(W\);
2. apply the input-side update virtually;
3. obtain

   \[
   W_i = WR_{\mathrm{in}};
   \]

4. replay the same minibatch at \(W_i\);
5. recompute the output-side signal;
6. compare stale and fresh output directions.

---

# 3. Staleness metrics

## 3.1 Raw generator cosine

For output-to-input coupling:

\[
s_{\mathrm{raw},o\rightarrow i}
=
\frac{
\left\langle
K_{\mathrm{in}}^{\mathrm{before}},
K_{\mathrm{in}}^{\mathrm{after}}
\right\rangle_F
}{
\left\|
K_{\mathrm{in}}^{\mathrm{before}}
\right\|_F
\left\|
K_{\mathrm{in}}^{\mathrm{after}}
\right\|_F
+
\epsilon
}.
\]

For input-to-output coupling:

\[
s_{\mathrm{raw},i\rightarrow o}
=
\frac{
\left\langle
K_{\mathrm{out}}^{\mathrm{before}},
K_{\mathrm{out}}^{\mathrm{after}}
\right\rangle_F
}{
\left\|
K_{\mathrm{out}}^{\mathrm{before}}
\right\|_F
\left\|
K_{\mathrm{out}}^{\mathrm{after}}
\right\|_F
+
\epsilon
}.
\]

Interpretation:

| Cosine | Meaning |
|---:|---|
| Near \(1\) | First-side update barely changes the second-side direction |
| \(0.5\)–\(0.9\) | Material but moderate direction change |
| Near \(0\) | Fresh direction is almost unrelated to stale direction |
| Negative | Fresh direction opposes the stale direction |

---

## 3.2 Final optimizer-direction cosine

Because `lie_ortho` changes the raw momentum direction, also compare the final
weight-space updates.

For output-first:

\[
D_{\mathrm{in}}^{\mathrm{stale}}
=
W_oA_{\mathrm{in}}^{\mathrm{stale}},
\]

\[
D_{\mathrm{in}}^{\mathrm{fresh}}
=
W_oA_{\mathrm{in}}^{\mathrm{fresh}}.
\]

Then log

\[
s_{\mathrm{step},o\rightarrow i}
=
\frac{
\left\langle
D_{\mathrm{in}}^{\mathrm{stale}},
D_{\mathrm{in}}^{\mathrm{fresh}}
\right\rangle_F
}{
\left\|
D_{\mathrm{in}}^{\mathrm{stale}}
\right\|_F
\left\|
D_{\mathrm{in}}^{\mathrm{fresh}}
\right\|_F
+
\epsilon
}.
\]

Repeat in the opposite order for

\[
s_{\mathrm{step},i\rightarrow o}.
\]

This metric is especially important because equal-angle orthogonalization can
magnify a modest raw-direction difference into a large final-step difference.

---

## 3.3 Relative direction change

Log

\[
M_{o\rightarrow i}
=
\frac{
\left\|
K_{\mathrm{in}}^{\mathrm{after}}
-
K_{\mathrm{in}}^{\mathrm{before}}
\right\|_F
}{
\left\|
K_{\mathrm{in}}^{\mathrm{before}}
\right\|_F+\epsilon
}.
\]

Likewise,

\[
M_{i\rightarrow o}
=
\frac{
\left\|
K_{\mathrm{out}}^{\mathrm{after}}
-
K_{\mathrm{out}}^{\mathrm{before}}
\right\|_F
}{
\left\|
K_{\mathrm{out}}^{\mathrm{before}}
\right\|_F+\epsilon
}.
\]

This captures both direction and magnitude changes.

---

# 4. Does freshness actually improve the loss?

A direction may change without producing a better update.

Construct two output-first candidates:

\[
W_{\mathrm{stale}}
=
R_{\mathrm{out}}
W
R_{\mathrm{in}}^{\mathrm{stale}},
\]

\[
W_{\mathrm{fresh}}
=
R_{\mathrm{out}}
W
R_{\mathrm{in}}^{\mathrm{fresh}}.
\]

Define the freshness advantage:

\[
P_{o\rightarrow i}
=
L(W_{\mathrm{stale}})
-
L(W_{\mathrm{fresh}}).
\]

Similarly,

\[
P_{i\rightarrow o}
=
L(W_{\mathrm{stale}})
-
L(W_{\mathrm{fresh}})
\]

for the reverse ordering.

Interpretation:

| Value | Meaning |
|---:|---|
| \(P>0\) | Fresh second-side direction gives a lower loss |
| \(P\approx0\) | Direction staleness is probably not important |
| \(P<0\) | Stale momentum may provide useful smoothing or regularization |

Evaluate each candidate on:

1. the minibatch used to construct the directions;
2. a held-out diagnostic minibatch.

The held-out result is more important because the fresh candidate has an
in-sample advantage on the batch used to recompute it.

---

# 5. Direction overlap and cancellation

Let the actual first-order weight-space directions be

\[
D_{\mathrm{out}}
=
A_{\mathrm{out}}W,
\]

\[
D_{\mathrm{in}}
=
WA_{\mathrm{in}}.
\]

Log their cosine:

\[
\rho_{\mathrm{oi}}
=
\frac{
\langle
D_{\mathrm{out}},
D_{\mathrm{in}}
\rangle_F
}{
\|D_{\mathrm{out}}\|_F
\|D_{\mathrm{in}}\|_F
+
\epsilon
}.
\]

Also log the joint movement ratio:

\[
r_{\mathrm{joint}}
=
\frac{
\|D_{\mathrm{out}}+D_{\mathrm{in}}\|_F^2
}{
\|D_{\mathrm{out}}\|_F^2
+
\|D_{\mathrm{in}}\|_F^2
+
\epsilon
}.
\]

Interpretation:

| Observation | Meaning |
|---|---|
| \(\rho_{\mathrm{oi}}<0\) | Input and output updates partially cancel |
| \(\rho_{\mathrm{oi}}\approx0\) | Approximately independent directions |
| \(\rho_{\mathrm{oi}}>0\) | Directions reinforce one another |
| \(r_{\mathrm{joint}}<1\) | Net movement is reduced by cancellation |
| \(r_{\mathrm{joint}}>1\) | Net movement is amplified by alignment |

Also log the separate first-order descent predictions:

\[
p_{\mathrm{out}}
=
-\langle G,D_{\mathrm{out}}\rangle_F,
\]

\[
p_{\mathrm{in}}
=
-\langle G,D_{\mathrm{in}}\rangle_F.
\]

A side with \(p<0\) is locally uphill, even if `lie_ortho` still assigns it a
full-angle update.

---

# 6. Finite-step interaction test

Evaluate four losses on the same deterministic diagnostic minibatch:

\[
L_0 = L(W),
\]

\[
L_o = L(R_{\mathrm{out}}W),
\]

\[
L_i = L(WR_{\mathrm{in}}),
\]

\[
L_{oi}
=
L(R_{\mathrm{out}}WR_{\mathrm{in}}).
\]

Define the finite interaction:

\[
I_{\mathrm{oi}}
=
L_{oi}
-
L_o
-
L_i
+
L_0.
\]

Interpretation:

| Value | Meaning |
|---:|---|
| \(I_{\mathrm{oi}}>0\) | Destructive interaction |
| \(I_{\mathrm{oi}}<0\) | Constructive interaction |
| \(I_{\mathrm{oi}}\approx0\) | Approximately additive updates |

This test uses the actual finite Cayley rotations and therefore captures:

- the bilinear term \(A_{\mathrm{out}}WA_{\mathrm{in}}\);
- model curvature;
- nonlinearity of the Cayley transform;
- finite-angle effects omitted by the first-order approximation.

---

# 7. True Gauss–Seidel oracle experiment

Run four controlled training arms.

## Arm A: current alternating baseline

- Write output on one step.
- Write input on the next step.
- Continue updating both momenta every step.
- Use the current best POET recipe.

## Arm B: two-pass stale control

Within one optimizer step:

1. run backward at \(W\);
2. construct both directions from \(W\);
3. optionally perform a second backward pass for compute matching;
4. apply output then input without recomputing the second-side direction.

Purpose:

> Control for the cost of an additional forward/backward pass without giving
> the second side a fresh direction.

## Arm C: true output-first Gauss–Seidel

Within one optimizer step:

1. forward/backward at \(W\);
2. construct and apply the output update;
3. obtain \(W_o=R_{\mathrm{out}}W\);
4. replay forward/backward at \(W_o\);
5. construct the input update from the fresh gradient;
6. apply the fresh input update.

## Arm D: true input-first Gauss–Seidel

Reverse Arm C:

1. compute and apply input update;
2. recompute gradient;
3. compute and apply fresh output update.

---

## Required controls

Match the following across Arms B–D:

- minibatch sequence;
- dropout/random-number state;
- optimizer momentum convention;
- learning-rate schedule;
- total tokens processed;
- total angle per side per token;
- realized

  \[
  \frac{\|\Delta W\|_F}{\|W\|_F};
  \]

- number of gradient observations inserted into each momentum state.

The decisive comparison is:

\[
\text{true fresh Gauss–Seidel}
\quad\text{versus}\quad
\text{two-pass stale control}.
\]

If true Gauss–Seidel wins, the advantage is caused by fresh second-side
re-evaluation rather than simply additional compute.

---

# 8. Two-scalar coordination oracle

Before changing either generator direction, estimate the best relative strength
of the two sides.

Let

\[
D_o=A_{\mathrm{out}}W,
\qquad
D_i=WA_{\mathrm{in}}.
\]

Consider the combined update

\[
D(\alpha,\beta)
=
\alpha D_o+\beta D_i.
\]

Define

\[
c=
\begin{bmatrix}
\langle G,D_o\rangle_F\\
\langle G,D_i\rangle_F
\end{bmatrix},
\]

and

\[
M=
\begin{bmatrix}
\|D_o\|_F^2
&
\langle D_o,D_i\rangle_F
\\
\langle D_o,D_i\rangle_F
&
\|D_i\|_F^2
\end{bmatrix}.
\]

Solve

\[
\min_{\alpha,\beta\ge0}
\quad
c^\top
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}
+
\frac{\lambda}{2}
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}^{\!\top}
M
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}.
\]

Initially, use this only as a logging oracle.

Log:

- \(\alpha^\star\);
- \(\beta^\star\);
- \(\alpha^\star/\beta^\star\);
- frequency with which one coefficient is zero;
- frequency with which the unconstrained optimum is negative;
- condition number of \(M\).

Interpretation:

| Oracle behavior | Implication |
|---|---|
| \(\alpha^\star\approx\beta^\star\) | Equal simultaneous angles are locally reasonable |
| Both positive but consistently unequal | Use side-specific angles |
| Usually one coefficient is zero | Adaptive side selection may outperform fixed parity |
| Unconstrained coefficient is frequently negative | One side's direction is stale or locally harmful |
| \(M\) is nearly singular | Strong overlap or gauge-like redundancy |

---

# 9. Momentum freshness ablation

The failed frozen-inactive-momentum experiment changed several effects at once.

Run the following controlled variants.

## Variant 1: full freshness

Current successful behavior:

\[
m_s
\leftarrow
\beta m_s
+
(1-\beta)g_s
\]

for both sides every optimizer step, even though only one side is written.

## Variant 2: decay only

For the inactive side:

\[
m_{\mathrm{inactive}}
\leftarrow
\beta m_{\mathrm{inactive}},
\]

but do not add its current gradient.

Purpose:

> Test whether maintaining the correct momentum clock is sufficient.

## Variant 3: buffered inactive gradients

Store inactive-side gradients but do not immediately insert them into momentum.

When the side becomes active, aggregate or average the buffered gradients and
perform one momentum update.

Purpose:

> Separate gradient sampling frequency from momentum update frequency.

## Variant 4: gradient freshness without write freshness

Update both momentum states from fresh gradients every step, but delay each
side's write according to the alternating schedule.

This is the current champion behavior and should remain the control.

---

# 10. Layer-phase synchronization test

Current global alternation may update the same side in every POET layer on the
same step.

Test whether this network-wide synchronization is harmful.

## Global parity

\[
\text{side}(t,\ell)
=
t\bmod2.
\]

Every layer writes the same side on a given step.

## Checkerboard parity

\[
\text{side}(t,\ell)
=
(t+\ell)\bmod2.
\]

Even and odd layers write opposite sides.

## Random fixed phase

Assign each matrix a fixed phase

\[
\phi_\ell\in\{0,1\},
\]

then use

\[
\text{side}(t,\ell)
=
(t+\phi_\ell)\bmod2.
\]

Required properties:

- every layer still alternates input/output;
- both momenta stay fresh;
- per-layer write frequency stays unchanged;
- no extra forward or backward pass is required.

If checkerboard or random phase improves training, some of the alternating gain
comes from avoiding a globally synchronized input/output update phase.

---

# 11. Optional symmetric Gauss–Seidel experiment

After determining whether output-first or input-first is better, test a
palindromic update.

For example:

\[
\frac{1}{2}\text{ output}
\rightarrow
\text{fresh full input}
\rightarrow
\frac{1}{2}\text{ fresh output}.
\]

Or the reverse:

\[
\frac{1}{2}\text{ input}
\rightarrow
\text{fresh full output}
\rightarrow
\frac{1}{2}\text{ fresh input}.
\]

The intermediate gradients must be recomputed.

Using the same stale generators for all three substeps does not test symmetric
Gauss–Seidel because fixed left and right matrix multiplications commute as
operators on \(W\).

---

# 12. Logging schema

For each sampled matrix, log the following.

## Metadata

- global step;
- transformer depth;
- module type:
  - attention Q;
  - attention K;
  - attention V;
  - attention output;
  - MLP up;
  - MLP gate;
  - MLP down;
- active side;
- update order;
- effective angle;
- learning rate.

## Staleness

- `raw_cos_out_to_in`;
- `raw_cos_in_to_out`;
- `step_cos_out_to_in`;
- `step_cos_in_to_out`;
- `relative_change_out_to_in`;
- `relative_change_in_to_out`.

## Loss probes

- `loss_base`;
- `loss_out_only`;
- `loss_in_only`;
- `loss_joint_stale`;
- `loss_joint_fresh_out_to_in`;
- `loss_joint_fresh_in_to_out`;
- `freshness_advantage_train_batch`;
- `freshness_advantage_heldout_batch`;
- `finite_interaction`.

## Direction geometry

- `norm_D_out`;
- `norm_D_in`;
- `cos_D_out_D_in`;
- `joint_movement_ratio`;
- `predicted_descent_out`;
- `predicted_descent_in`;
- `predicted_descent_joint`.

## Angle oracle

- `oracle_alpha`;
- `oracle_beta`;
- `oracle_alpha_beta_ratio`;
- `oracle_one_hot`;
- `oracle_unconstrained_negative`;
- `direction_gram_condition_number`.

## Momentum

- raw-gradient-to-momentum cosine;
- stale-momentum-to-fresh-momentum cosine;
- momentum-to-final-orthogonalized-direction cosine;
- inactive-side gradient norm;
- inactive-side momentum norm;
- active-side gradient norm;
- active-side momentum norm.

---

# 13. Sampling strategy

Do not run the expensive probes on every matrix and every step.

Sample:

- one early transformer block;
- one middle transformer block;
- one late transformer block.

Within each block, sample:

- attention output;
- one of Q/K/V;
- MLP up or gate;
- MLP down.

Suggested frequency:

- every 250–500 optimizer steps;
- more frequently near:
  - the end of warmup;
  - the start of learning-rate decay;
  - the point where POET begins separating from AdamW or Muon;
  - the final 10–20% of training.

Use deterministic diagnostic minibatches so that changes over training are
comparable.

---

# 14. Decision table

| Observation | Most likely mechanism | Next action |
|---|---|---|
| Low stale/fresh cosine and fresh candidate wins held-out loss | Genuine second-side staleness | Pursue true Gauss–Seidel or predictor–corrector updates |
| High stale/fresh cosine but positive finite interaction | Finite-step destructive interaction | Reduce joint angle, use trust-region allocation, or serialize writes |
| Strongly negative input/output direction cosine | Cancellation | Jointly allocate angles or enforce non-destructive combination |
| Two-scalar oracle is usually one-hot | Only one side is locally useful at a time | Adaptive side selection |
| Oracle prefers two positive but unequal coefficients | Both sides useful with asymmetric strength | Per-side adaptive angles |
| Staleness is small but momentum ablations differ strongly | Temporal filtering is the main benefit | Improve momentum estimator rather than update order |
| Checkerboard phase beats global parity | Network-wide phase synchronization is harmful | Stagger side phases by layer |
| Output-first consistently beats input-first | Systematic order asymmetry | Use output-first Gauss–Seidel |
| Input-first consistently beats output-first | Systematic order asymmetry | Use input-first Gauss–Seidel |
| Ordering preference changes by module type | Layer-type-dependent coupling | Assign order by module class |
| Ordering preference changes over training | Time-dependent coupling | Schedule the ordering or use an online selection rule |

---

# 15. Recommended execution order

## Phase 1: diagnostic-only

Implement without changing training:

1. stale/fresh raw-direction cosine;
2. stale/fresh final-step cosine;
3. input/output update overlap;
4. finite-step four-loss interaction;
5. two-scalar angle oracle.

Goal:

> Identify whether the main issue is staleness, interference, cancellation, or
> angle allocation.

## Phase 2: short checkpoint forks

From the same checkpoint, run:

1. current alternating;
2. two-pass stale control;
3. true output-first Gauss–Seidel;
4. true input-first Gauss–Seidel.

Use short continuations to compare:

- immediate validation loss;
- predicted versus realized descent;
- stability;
- direction coherence.

## Phase 3: full training experiment

Only promote the mechanism supported by the diagnostic.

Possible promoted variants:

- true Gauss–Seidel;
- adaptive side selection;
- per-side angle allocation;
- checkerboard layer phase;
- symmetric Gauss–Seidel;
- predictor–corrector approximation.

---

# 16. Primary success criteria

A coordination mechanism is promising when it produces all of the following:

1. better held-out candidate loss than the stale simultaneous direction;
2. improved validation loss at matched tokens;
3. matched or controlled effective weight-space movement;
4. no loss of inactive-side momentum freshness;
5. no stability regression near the current angle ceiling;
6. improvement across at least two random seeds;
7. consistent benefit across early, middle, and late training.

The key mechanistic result to establish is:

\[
\boxed{
\text{Does recomputing the second-side direction after updating the first side
produce a materially better held-out update?}
}
\]

If yes, current alternating is likely a low-cost approximation to a stronger
Gauss–Seidel-style two-sided POET optimizer.

---

# 17. Implemented Tier-0 instrumentation (`poet_coord/*`)

This section documents the diagnostics that are **actually wired** (Phase 1,
diagnostic-only — no training change). They are computed in the optimizer's
pre-`step` hook and logged to W&B, on the live champion run. No second backward,
no control arm needed for the first read.

- Pure math: [src/diag/poet_coordination_diag.py](/lustre/fast/fast/zqiu/slm-research/src/diag/poet_coordination_diag.py)
- Install / state lookup / W&B: [src/patches/poet_coordination_log.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_coordination_log.py)
- Enable: `SLM_POET_COORD_DIAG=1` (interval `SLM_POET_COORD_DIAG_INTERVAL`,
  default 250). Inert otherwise (registered in `_ALWAYS_ON_PATCHES`).

## 17.1 What is read, and when

At each `optimizer.step` that lands on the logging interval, for each sampled
two-sided layer the hook reads, **before** the optimizer consumes them:

- both sides' Lie momentum \(m_{\mathrm{out}}, m_{\mathrm{in}}\) (`lie_m`, which
  persists across the per-step fold) from optimizer state;
- both sides' **fresh** skew-tangent gradient \(g_{\mathrm{out}}, g_{\mathrm{in}}\)
  at the current weight (`oft_R.main_grad`, falling back to `.grad`);
- the layer weight \(W\).

Both momenta are read regardless of which side is *written* this step, so the
metrics have the **same definition on the alternating, simultaneous, and frozen
arms** — an apples-to-apples read across the 3-arm comparison.

Each side's quantities are stored in *vec form*: \(r_s\) blocks
\(\times\,n_{\mathrm{elems}}=b_s(b_s-1)/2\) strictly-upper-triangular entries.

## 17.2 Derived quantities

**Realized generator direction.** The optimizer writes
\(\operatorname{ortho\_c}\cdot\operatorname{orthogonalize}(-m_s)\); the diagnostic
uses the same Muon Newton–Schulz transform (so it reflects the direction the
optimizer *would* write), dropping the common \(\operatorname{ortho\_c}\cdot
\mathrm{lr}\) scale (it cancels in every ratio below):

\[
A_s = \operatorname{orthogonalize}\!\big(\operatorname{vec\_to\_skew}(-m_s)\big),
\qquad s\in\{\mathrm{out},\mathrm{in}\}.
\]

**W\_perm frame.** POET's generators are block-diagonal only in the un-permuted
frame, so \(W\) is mapped back through the inverse forward permutations
(`w_perm_frame`):

\[
W_{\mathrm{perm}} = W[\,\psi_{\mathrm{out}}^{-1}\,][:,\ \psi_{\mathrm{in}}^{-1}\,].
\]

**Weight-space side directions** (`side_directions`; one `bmm` + one `einsum`,
never materializing the dense block-diagonal):

\[
D_{\mathrm{out}} = \operatorname{blockdiag}(A_{\mathrm{out}})\,W_{\mathrm{perm}},
\qquad
D_{\mathrm{in}} = W_{\mathrm{perm}}\,\operatorname{blockdiag}(A_{\mathrm{in}}).
\]

\(\cos(D_{\mathrm{out}},D_{\mathrm{in}})\) is permutation-invariant, so the
overlap geometry is the same as in the forward frame.

## 17.3 The metric → mechanism map

| W&B key (`poet_coord/<layer>/…`, plus `poet_coord/_mean/…`) | Definition / how computed | Mechanism it arbitrates | The read |
|---|---|---|---|
| `mom_cos_out`, `mom_cos_in` | \(\dfrac{\langle m_s, g_s\rangle_F}{\lVert m_s\rVert_F\,\lVert g_s\rVert_F}\), reduced over the whole \((r_s, n_{\mathrm{elems}})\) tensor (one scalar/side). The \(\sqrt2\) skew↔vec factor cancels; zero input → 0. | **staleness / SNR** (fact #5) | **Empirically (qjapxj18) the champion reads ≈0 (mildly negative, \(-0.2\to 0\) as LR decays), NOT \(\gtrsim 0.8\)** — the per-step rotation gradient is near-white, so the EMA *averages noise* rather than tracking a stable direction (and the mild negative tilt is a per-step *overshoot* fingerprint of eff∠ 0.016). Healthy = small-and-stable; the discriminator is the frozen arm, whose reactivated side should decohere further / lose momentum norm. |
| `cos_D_out_D_in` | \(\dfrac{\langle D_{\mathrm{out}}, D_{\mathrm{in}}\rangle_F}{\lVert D_{\mathrm{out}}\rVert_F\,\lVert D_{\mathrm{in}}\rVert_F}\) | **gauge-redundancy** (champion > simultaneous) | persistently \(\lvert\cos\rvert>0.3\) (esp. attn-out / MLP-down) → a matched-\(\lVert\Delta W\rVert\) simultaneous step over-spends the redundant direction. \(\cos\approx 0\) **falsifies** redundancy → look elsewhere. **(Empirically ≈0 all run, qjapxj18 — falsified; see §17.6.)** |
| `cos_D_out_D_in_raw` | same, but \(A=\text{raw }(-m)\) (no NS) | is the decorrelation **intrinsic or NS-induced** | raw correlated but orthogonalized ≈0 → Muon whitening decorrelates the sides; both ≈0 → intrinsic |
| `r_cross` | \(\dfrac{\lVert A_{\mathrm{out}}WA_{\mathrm{in}}\rVert_F}{\lVert A_{\mathrm{out}}W\rVert_F+\lVert WA_{\mathrm{in}}\rVert_F}\), via \(\operatorname{blockdiag}(A_{\mathrm{out}})\,D_{\mathrm{in}}\) (one extra block-matmul) | **finite-step coupling** the first-order overlap can't see | \(\sim\) eff∠ (≈0.016) ⇒ decoupled at finite order too; materially larger / growing ⇒ real bilinear coupling (the channel the alternating win could flow through) |
| `gram_cond` | condition number of \(M=\begin{bmatrix}\lVert D_{\mathrm{out}}\rVert^2 & \langle D_{\mathrm{out}},D_{\mathrm{in}}\rangle\\ \langle D_{\mathrm{out}},D_{\mathrm{in}}\rangle & \lVert D_{\mathrm{in}}\rVert^2\end{bmatrix}\), via the analytic 2×2 eigenvalues \(\lambda_\pm=\tfrac{a+b}{2}\pm\sqrt{(\tfrac{a-b}{2})^2+c^2}\) | same | routinely \(5\text{–}50\times\) confirms a near-singular 2-direction subspace |
| `r_joint` | \(\dfrac{\lVert D_{\mathrm{out}}+D_{\mathrm{in}}\rVert_F^2}{\lVert D_{\mathrm{out}}\rVert_F^2+\lVert D_{\mathrm{in}}\rVert_F^2}\) | overlap sign | \(<1\) cancellation, \(=1\) orthogonal, \(>1\) reinforcement |
| `norm_D_out`, `norm_D_in` | \(\lVert D_{\mathrm{out}}\rVert_F,\ \lVert D_{\mathrm{in}}\rVert_F\) | relative per-side movement | which side actually moves \(W\) more |

`poet_coord/_mean/<metric>` is the mean over all sampled layers; the per-layer
keys keep the depth/module-type breakdown (attn-out and MLP-down are the
expected redundancy hot spots).

## 17.4 Cost and sampling

Per sampled layer: two Frobenius cosines (free) + one Newton–Schulz
orthogonalize and one `bmm`/`einsum` per side (\(\sim\) one extra matmul). **No
extra forward or backward.** Sampling: the wanted projections (q/v/fc1/fc2 …)
that carry **both** `oft_R_in` and `oft_R_out`, capped at `max_targets`, every
`SLM_POET_COORD_DIAG_INTERVAL` steps.

## 17.5 Tier-1 additions (wired) and what is still deferred

**Wired** (added after the `qjapxj18` read, to probe whether the sides are *truly*
decoupled or merely first-order orthogonal — one extra block-matmul each, no backward):

- `cos_D_out_D_in_raw` — the overlap from the **raw** \(-m\) directions (no NS). If the
  raw cos is correlated but `cos_D_out_D_in` (orthogonalized) is ≈0, the Muon whitening
  is what decorrelates the two sides; if raw is also ≈0, the decorrelation is intrinsic.
- `r_cross` \(=\lVert A_{\mathrm{out}}WA_{\mathrm{in}}\rVert_F/(\lVert A_{\mathrm{out}}W\rVert_F+\lVert WA_{\mathrm{in}}\rVert_F)\)
  — the finite bilinear cross-term magnitude. First-order orthogonality
  (`cos_D_out_D_in`≈0) does **not** imply this is zero; it is the coupling channel the
  overlap metric is blind to. Expected \(\sim\) eff∠ (≈0.016) if the sides are decoupled
  at finite order too; materially larger / growing ⇒ real finite-step coupling.

**Still deferred** (these need the ambient \(G=\partial L/\partial W\) and/or extra
forward passes, so they cannot reuse the optimizer-state hook):

- Cross-term **loss alignment** \(-\langle G, A_{\mathrm{out}}WA_{\mathrm{in}}\rangle\) —
  needs \(G\) (the hook only has the skew-tangent grads \(K=\operatorname{skew}(GW^\top)\)).
- Finite four-loss interaction \(I_{oi}=L_{oi}-L_o-L_i+L_0\) (§6) — needs a diagnostic
  minibatch and four forward passes.
- Weight-only staleness split \(\cos(W^\top G,\ W_o^\top G)\) reusing the same \(G\).

## 17.6 First champion read (`qjapxj18`, 60m / lr4e-3 / scale0.5 / c8)

Full-training read of the champion (`coord_champion`, W&B `qjapxj18`), 41 diag
points step 1000 → 8250:

- `cos_D_out_D_in` \(\in[-0.002, +0.002]\) for the **entire run**; `gram_cond`
  \(\approx 1.25\); `r_joint` \(\approx 1.000\). The two sides' weight-space
  directions are **orthogonal throughout training** →
  **gauge-redundancy is falsified** (not just at plateau). The gauge-decorrelation
  lever is therefore off the table — there is no redundant subspace to decorrelate.
- `mom_cos_{out,in}` start mildly negative (\(\sim-0.12\) to \(-0.23\)) and decay
  to \(\approx 0\) by step ~8000 as the cosine LR approaches its floor. The per-step
  rotation gradient is **near-white** (low SNR all run), so the persistent EMA is
  doing genuine **noise-averaging** — which is why freezing it regresses to 4.22
  (`au92x0pj`). The mild negative tilt is a small per-step **overshoot** signature
  consistent with eff∠ 0.016 being the hot edge.

**Refined verdict:** the alternating advantage is **temporal / momentum
(noise-averaging + fresh-W re-evaluation)**, *not* spatial overlap/cancellation.
This also reframes POET_dev's "Gauss–Seidel coupling" attribution: the coupling is
temporal, not a spatial redundancy between the two directions. Next levers are
momentum-estimator ones (the `alternate_every` averaging-window sweep; possibly a
slightly cooler or `mom_cos`-gated angle), not direction-geometry.
