#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH--time=24:00:00 

source path.sh

# Add `--dry_run` below for a config-only sanity check.
python run.py \
    --stages all \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    "$@"
