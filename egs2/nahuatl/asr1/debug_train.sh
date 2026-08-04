#!/usr/bin/env bash
# Submit a 1-epoch / 20-step debug training run via SLURM.
# Adjust --account and partition as needed for your cluster.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --job-name=nahuatl-debug
#SBATCH --output=%x_%j.log

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cd "$RECIPE_DIR"
source path.sh

max_epoch=1 num_iters_per_epoch=20 log_interval=1 stage=3 stop_stage=3 bash run.sh \
  2>&1 | tee debug_train_live.log
