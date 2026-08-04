#!/usr/bin/env bash
# Full training run (stages 10-11: collect_stats + train) via SLURM.
# Adjust --account and partition as needed for your cluster.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=nahuatl-train
#SBATCH --output=%x_%j.log

RECIPE_DIR=/work/nvme/bbjs/clin10/nahuatl_asr/espnet/egs2/nahuatl/s2t1
cd "$RECIPE_DIR"
source path.sh

bash run.sh --stage 10 --stop_stage 11 \
    2>&1 | tee full_train_live.log
