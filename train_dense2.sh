#!/bin/bash

# Uncomment the line below if you want the script to stop immediately if a command fails
# set -e

echo "Starting sequential training runs..."

echo "2/6: Running Qwen3..."
bash scripts/train_bakeoff_600m.sh qwen3 cluster=h100_de wandb.project=slm-arch-dense

echo "3/6: Running DeepSeek Dense..."
bash scripts/train_bakeoff_600m.sh deepseek_v3_dense cluster=h100_de wandb.project=slm-arch-dense


echo "All training runs completed!"
