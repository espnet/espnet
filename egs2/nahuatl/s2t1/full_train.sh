#!/usr/bin/env bash
# Full training run (stages 3-11: format wav.scp -> collect_stats -> train) via SLURM.
# Stage 1 (local/data.sh, CPU data prep) is expected to have been run beforehand
# on a login node; this job covers the wav formatting, stats, and GPU training.
# Adjust --account and partition as needed for your cluster.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=nahuatl-train
#SBATCH --output=%x_%j.log

# pipefail so a failure in run.sh propagates through the `| tee` pipeline and is
# reflected in the job's exit status (otherwise tee's success would mask it).
set -o pipefail

# SLURM copies the batch script to a spool dir, so ${BASH_SOURCE[0]} is unreliable
# under sbatch. Use SLURM_SUBMIT_DIR (the dir sbatch was launched from) when set,
# and fall back to the script's own location for plain `bash full_train.sh`.
RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"
source path.sh

bash run.sh --stage 3 --stop_stage 11 \
    2>&1 | tee full_train_live.log
