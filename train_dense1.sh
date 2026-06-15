#!/bin/bash

# Uncomment the line below if you want the script to stop immediately if a command fails
# set -e

echo "Starting sequential training runs..."

echo "5/6: Running MiniCPM..."
bash scripts/train_bakeoff_600m.sh minicpm cluster=h100_de wandb.project=slm-arch-dense

echo "6/6: Running Llama3..."
bash scripts/train_bakeoff_600m.sh llama3 cluster=h100_de wandb.project=slm-arch-dense

echo "All training runs completed!"
