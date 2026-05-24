# POET Cayley cache GPU smoke runbook

Run on the cluster after Tasks 1–10 are merged. Mirrors spec §13 + §15.

## Step 1: Single-GPU unit-test parity

```bash
cd /lustre/fast/fast/zqiu/slm-research
pytest tests/unit/test_poet_cache.py -v
```

Expected: all parity tests PASS, including
`test_forward_none_mode_matches_upstream_poet_linear`,
`test_mode_b_single_microbatch_parity_with_none`,
`test_mode_b_K_microbatch_accumulation_parity_with_none`,
`test_mode_a_K_microbatch_parity_with_none`,
`test_compute_cayley_matches_upstream_get_weight_poet`.

## Step 2: 2-rank DDP smoke (spec §10 acceptance for Mode A under DDP)

Driving a DDP smoke from pytest requires a torchrun harness. The simplest
path is to write a small standalone driver under `tools/poet_ddp_smoke.py`
that:

1. Initializes Megatron's `parallel_state` with `data_parallel_size=2`.
2. Builds two ranks worth of identical CachedPOETLinear modules in
   `cached_fwd_bwd` mode (same seed, different `oft_R.main_grad`
   destinations because the buffer is allocated per rank).
3. Splits a single batch across both ranks and runs forward+backward
   with K=2 micro-batches per rank.
4. Calls `_flush_poet_caches_for_step()`.
5. Compares rank 0's `oft_R.main_grad` against a single-rank reference
   over the full unsplit batch.

Pass criterion: `oft_R.main_grad` matches the single-rank reference
within `atol=1e-5` (fp32) / `1e-2` (bf16).

If the smoke FAILS, the diagnostic is whether `_sync_oft_R_grads_across_dp`
is producing the expected averaged main_grad. Print pre-sync and
post-sync `main_grad` on each rank to localize.

## Step 3: 1k-step bf16 training-loss parity smoke

Launch a 1k-step Qwen3-600M run (or smallest available scale) three
times: `cache_mode=none`, `cache_mode=cached_fwd`, `cache_mode=cached_fwd_bwd`.
Same seed, same data, same hyperparams.

Pass criterion (per spec §15):
- Loss curve diff `|loss_cached − loss_none|` per step `< 1e-2`
  throughout the 1000 steps.
- Wall-clock per step in `cached_fwd_bwd` is measurably faster than
  `none`. Target: cayley-fraction × `(K-1)/K` improvement.

If wall-clock is much worse than target, profile and check:
- Are R-block tensors being freed/re-allocated each cycle? They should
  be reused — only their leaves are re-created on cache miss.
- Is `_compute_cayley` triggering recompiles? Run with
  `TORCH_LOGS=recompiles python ...` to verify (it shouldn't, because
  the wrapper is plain Python — see Task 3 design note).
- Is the manual all-reduce in `_sync_oft_R_grads_across_dp` actually
  packing into one flat buffer? (Profile with NCCL traces.)

If perf is in the right ballpark but the wrapper Python overhead is
visible in profiles, revisit the Task 3 decision and add
`@torch.compile(fullgraph=True)` to `_compute_cayley`.

## Step 4: Update CHANGELOG

After steps 1–3 pass, append a CHANGELOG entry recording the realized
speedup and any deviation from the planned design.
