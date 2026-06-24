# pion

Pion optimizer (`src/optim/_pion.py`, vendored from Sphere-AI-Lab/pion) — a
spectrum-preserving optimizer via orthogonal equivalence transformation. Matrix
weights are rotated by two Lie generators `W <- W exp(A_in) + exp(A_out) W - W`
(truncated matrix exponential); a chained Megatron AdamW drives the rest.

- **Scope:** single GPU (DP=TP=PP=1). The builder (`src/optim/pion.py`) raises on
  tensor parallelism, pipeline parallelism, the distributed optimizer, or fp16.
- **Param routing:** 2-D non-embedding/output weights → Pion; embeddings,
  lm_head, norms, biases → chained AdamW (`ChainedOptimizer`, like POET).
- **Fused (default):** qkv split per-head and fc1 up/gate split happen INSIDE the
  optimizer. The `model_unfuse_linears` patch is listed but no-op unless
  `base.model.unfuse_qkv`/`unfuse_fc1` are set — see `pion_unfused` for the
  opt-in unfused variant.
- **LR:** one `optim.lr` for both sides (the Pion side is RMS-scaled internally by
  `pion_rms*sqrt(m*n)`).
- **Defaults:** `pion_momentum=transported_ambient_ambient`,
  `pion_update_side=alternate`, `pion_scaling=rms`, `pion_rms=0.2`,
  `pion_degree=2`, betas `(0.9, 0.95)` — from `opt_llama_60M_pion.sh`.
- **Wiring:** `--optimizer adam --slm-optimizer pion`, rerouted by the
  `pion_optimizer_setup` patch.

Run on the 60m dev model with `scripts/train_pion_dev.sh`; sweep LR with
`scripts/sweep_pion_lr.sh`.
