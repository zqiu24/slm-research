# muon_kimi

Muon variant that runs the vendored Kimi/Moonlight single-process optimizer
(`src/optim/_kimi_muon.py`, adapted from KellerJordan/Muon, MIT) instead of the
Megatron-Core / emerging_optimizers `TensorParallelMuon` used by `muon_hybrid`.

- **Scope:** single GPU (DP=TP=PP=1). The builder
  (`src/optim/muon_kimi.py`) raises on tensor parallelism, pipeline parallelism,
  the distributed optimizer, or fp16.
- **Param routing:** 2-D non-embedding/output weights → Muon; embeddings,
  lm_head, norms, biases → internal AdamW.
- **LR:** one `optim.lr` for both sides (the Muon side is RMS-scaled by
  `0.2*sqrt(max(d_out,d_in))` internally), unlike `muon_hybrid`'s split LRs.
- **Wiring:** `--optimizer adam --slm-optimizer muon_kimi`, rerouted by the
  `muon_kimi_optimizer_setup` patch.

Run on the 60m dev model with `scripts/train_muon_kimi_dev.sh`.
