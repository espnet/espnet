#!/usr/bin/env bash
# Full training run (stages 10-11: collect_stats + train) via SLURM.
# Adjust --account and partition as needed for your cluster.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=nahuatl-train
#SBATCH --output=%x_%j.log

# SLURM copies the batch script to a spool dir, so ${BASH_SOURCE[0]} is unreliable
# under sbatch. Use SLURM_SUBMIT_DIR (the dir sbatch was launched from) when set,
# and fall back to the script's own location for plain `bash full_train.sh`.
RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"
source path.sh

bash run.sh --stage 10 --stop_stage 11 \
    2>&1 | tee full_train_live.log
