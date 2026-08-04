#!/usr/bin/env bash
# Submit a 1-epoch / 20-step debug training run via SLURM.
# Adjust --account and partition as needed for your cluster.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --job-name=nahuatl-debug
#SBATCH --output=%x_%j.log

# pipefail so a failure in run.sh propagates through the `| tee` pipeline and is
# reflected in the job's exit status (otherwise tee's success would mask it).
set -o pipefail

# SLURM copies the batch script to a spool dir, so ${BASH_SOURCE[0]} is unreliable
# under sbatch. Use SLURM_SUBMIT_DIR (the dir sbatch was launched from) when set,
# and fall back to the script's own location for plain `bash debug_train.sh`.
RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"
source path.sh

# s2t.sh stage 11 = S2T training (stages 1-10 are data prep / collect_stats)
max_epoch=1 num_iters_per_epoch=20 log_interval=1 \
    bash run.sh --stage 11 --stop_stage 11 \
    2>&1 | tee debug_train_live.log
