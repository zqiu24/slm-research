#!/usr/bin/env bash
# muon_kimi HP grid sweep — cosine scheduler.
# Run on one node (16 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_muon_kimi_grid.sh
#
# Baseline being tuned: muon_kimi (vendored Kimi/Moonlight Muon on 2D attn/MLP
# weights, internal AdamW on embeddings/norms/head) — the best NON-POET run at
# 60m/40tpp (val/loss 3.5321 @ lr 1e-3, `of4bakqd`). POET just overtook it
# (3.5231); this checks whether muon_kimi itself is under-tuned (esp. on LR — the
# POET grid found the baseline LR was too cold).
#
# Launcher = scripts/train_muon_dev.sh, which reproduces the cohort exactly:
# experiment=optim/muon_kimi, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024,
# mbs 128, transformer_impl=local, tie_embeddings=false. NOTE: muon_kimi uses ONE
# base optim.lr for BOTH the Muon side (internally scaled by 0.2*sqrt(max d_out,d_in))
# and the internal AdamW side — so optim.lr is fully coupled, like POET's.
#
# SWEPT (4 x 2 x 2 = 16):
#   optim.lr               in {5e-4, 1e-3, 2e-3, 3e-3}   (1e-3 = baseline; probe hotter)
#   optim.muon_momentum    in {0.95, 0.98}               (0.95 = default)
#   scheduler.min_lr_ratio in {0.1, 0.01}                (0.1 = default; 0.01 = POET-favored deep floor)
# Held at the muon_kimi defaults: muon_use_nesterov=true, muon_num_ns_steps=5,
# weight_decay=0.1, adam betas [0.9,0.95].
#
#   name              lr      mom    min_lr   note
#   mk_lr10_m95_f1    1e-3    0.95   0.1      = BASELINE (reproduces of4bakqd, val 3.5321)
#   mk_lr05_m95_f1    5e-4    0.95   0.1      colder LR
#   mk_lr20_m95_f1    2e-3    0.95   0.1      hotter LR
#   mk_lr30_m95_f1    3e-3    0.95   0.1      hottest LR
#   (× muon_momentum {0.95, 0.98} × min_lr_ratio {0.1, 0.01})

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

PREFIX="mk"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.0005 0.001 0.002 0.003); LTAGS=(05 10 20 30)
MOMS=(0.95 0.98);               MTAGS=(95 98)
FLOORS=(0.1 0.01);              FTAGS=(1 01)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  for j in "${!MOMS[@]}"; do
    mom="${MOMS[$j]}"; mt="${MTAGS[$j]}"
    for k in "${!FLOORS[@]}"; do
      fl="${FLOORS[$k]}"; ft="${FTAGS[$k]}"
      name="${PREFIX}_lr${lt}_m${mt}_f${ft}"
      echo "### ${name}: lr=${lr} muon_momentum=${mom} min_lr_ratio=${fl}"
      codexlog "$name" scripts/train_muon_dev.sh \
        optim.lr="$lr" optim.muon_momentum="$mom" scheduler.min_lr_ratio="$fl" \
        experiment.name="$name"
    done
  done
done

echo "=== muon_kimi grid sweep complete (16 runs) ==="
