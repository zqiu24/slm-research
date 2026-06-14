# Architecture-family bake-off @ 600M (2026-06)

**Question.** Which base architecture should slm-research pretrain from
scratch: DeepSeek-V3 (MLA + MoE + MTP), Qwen3-Next-style (GatedDeltaNet
hybrid + MoE), or Nemotron-H-style (Mamba2 hybrid, dense)? Control:
existing `qwen3` dense 600M rung.

**Design.** One run per family, seed 42; everything else frozen:
`training_regime=ablation_40x` (24B tokens), `scheduler=wsd`,
`experiment=optim/adam`, GBS 1024, seq 4096, dataset
`nemotron_cc_v2_llama31_8b` (manifest-frozen tokenizer). All scale files
declare `non_embedding_params: 600_000_000` → identical `--train-samples`
and a shared GPTDataset cache. The two MoE families share the DeepSeek
router recipe (sigmoid, seq_aux_loss 1e-4, expert bias, topk 4/16) so the
comparison isolates the mixer/backbone.

| family | scale | total non-emb | active | entrypoint |
|---|---|---|---|---|
| qwen3 (control) | 600m | ~600M | =total | gpt |
| deepseek_v3 | 600m_deepseek_v3 | 592.1M | ~252M | gpt |
| deepseek_v3_dense (opt.) | 600m_deepseek_v3_dense | 604.3M | =total | gpt |
| qwen3_next | 600m_qwen3_next | 594.9M | ~241M | gpt |
| nemotron_h | 600m_nemotron_h | 604.8M | =total | mamba |

`deepseek_v3_dense` is an **optional** dense ablation (MLA + MTP identical to
`deepseek_v3`, MoE replaced by a dense SwiGLU FFN). It isolates the value of
sparsity at equal total non-embedding params. Not in the default 4-family
sweep; add it with `FAMILIES="... deepseek_v3_dense"` or run it on its own.

**Known asymmetries (accepted).** Hidden size differs (1280 for
nemotron_h/control vs 1024 for the MoE families) → tied-embedding counts
differ (embeddings sit outside the budget unit per SPEC §1.3). MoE families
have ~2.4x fewer active params per token — recorded, not equalized. LR is
the optim/adam default for every family (per-family LR tuning is a
follow-up sweep, not part of the controlled comparison). DeepSeek keeps its
MTP head (family identity); its `lm loss` is the comparison metric, not the
MTP auxiliary loss. qwen3_next approximates the published model on its
full-attention layers (no per-head output gate, standard rather than
zero-centered RMSNorm — no native Megatron flags for either).

**Launch.**
    bash scripts/train_bakeoff_600m.sh <family> cluster=<cluster>

**Decision metrics, in order.**
1. Validation `lm loss` at 24B tokens (primary; from the shared W&B project,
   runs auto-grouped by config identity).
2. Loss-vs-tokens curve over the final 20% (slope still healthy? crossovers?).
3. Train throughput (tokens/s/GPU, `--log-throughput`) and peak reserved
   memory — the efficiency term that scales to the 1.2B/2.4B promotions.
4. Stability: no loss spikes/divergence, grad-norm sane, (MoE) aux loss and
   router load-balance healthy.
5. Qualitative: Megatron parallelism maturity + Megatron-Bridge export path
   at promotion scale.

**Decision rule.** Best validation loss wins unless within seed noise of the
runner-up (use the champion ladder's seed-variance band; if inside it, rerun
the tied families at seeds 43, 44) — ties break on throughput, then
stability. Record the verdict + W&B links in the Results section below, then
promote the winner: realize `1_2b_<winner>.yaml` with tools/size_check.py
and run the 1.2B gate.

**Results.** _(fill after runs)_

| family | val loss @24B | tok/s/GPU | peak mem | verdict |
|---|---|---|---|---|
| qwen3 (control) | | | | |
| deepseek_v3 | | | | |
| deepseek_v3_dense (opt.) | | | | |
| qwen3_next | | | | |
| nemotron_h | | | | |
