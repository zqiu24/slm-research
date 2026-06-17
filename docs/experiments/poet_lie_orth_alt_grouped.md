# poet_lie_orth_alt_grouped

Grouped POETX over MoE experts on the champion `lie_ortho` recipe.

Identical to [`poet_lie_orth_alt`](poet_lie_orth_alt.md) (integrated alternating
POETX, lie_ortho, both-momenta, head-OFF, lr 3e-3, c=8, distributed) but with
`optim.poet.group_experts: true`. All E experts in each MoE layer share a single
batched `GroupedPOETXLinear` (Tasks 2–6 of the grouped-POETX plan) rather than E
independent `POETXLinear` layers.

- **Design + plan:** `docs/superpowers/plans/2026-06-17-grouped-poetx-over-experts.md`
- **Design spec:** `docs/superpowers/specs/2026-06-17-grouped-poetx-over-experts-design.md`
- **Baseline:** `poet_lie_orth_alt` (per-expert POETX, val/loss ≈ 3.47–3.55 range)
- **Throughput target:** batched block-sparse backward replaces E independent
  M-builds with one; merge stays the verified per-expert Cayley fold (2.6%
  overhead). Win expected on the expert GEMM / `M` ops that dominate forward/backward.
- **NOTE:** `group_experts` is orthogonal to `moe.grouped_gemm`. The config
  keeps `grouped_gemm=false`; the existing POET guard rejects poet+grouped_gemm.

## Results

<!-- GPU steps 7–9 are the user's to run on the cluster. Fill in after the A/B. -->

### Step 7 — Profile gate (USER-RUN)

```bash
env POET_PROFILE_STEP=20 POET_PROFILE_TORCH=1 \
  bash scripts/train_deepseek_poet.sh full \
  training.global_batch_size=8 training.micro_batch_size=1 training.log_interval=1
```

Expected: expert GEMM / `M` rows dominate the `torch.profiler top ops` table.

### Step 8 — 1-GPU smoke (USER-RUN)

```bash
codexlog poetx_grouped_smoke bash scripts/train_deepseek_poet.sh dev optim.poet.group_experts=true training.log_interval=1
```

Acceptance: builds; `[POET] replaced N` logs the grouped experts; finite loss;
loss at step ~10 matches the non-grouped `dev` run within fp noise.

### Step 9 — 8-GPU throughput + loss A/B (USER-RUN)

```bash
codexlog poetx_grouped_full bash scripts/train_deepseek_poet.sh full optim.poet.group_experts=true
codexlog poetx_baseline_full bash scripts/train_deepseek_poet.sh full          # per-expert POET baseline
```

Acceptance: grouped TFLOP/s materially > the 4.2 baseline; lm-loss trajectory
within noise of the per-expert POET run over the same steps.

| Run | TFLOP/s | val loss | notes |
|-----|---------|----------|-------|
| poetx_baseline_full (per-expert) | — | — | |
| poetx_grouped_full (grouped) | — | — | |
