#!/bin/bash
#SBATCH --job-name=must_c_st_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --qos=bbjs-delta-gpu
#SBATCH --output=/work/hdd/bbjs/sjin2/espnet/training_log/must_c_st_infer_%j.log
#
# Full evaluation: decode both egs2 test sets and score them.
#
#   INFERENCE_CONFIG=conf/inference_epoch41.yaml sbatch egs3/must_c/st/submit_infer.sh
#
# 3,241 utterances (tst-COMMON 2,641 + tst-HE 600) at beam 10, ~2.8 s each in a
# single process, so ~2.5 h; 6 h is requested for headroom. inference.yaml sets
# `parallel: env: local`, which is ONE shard regardless of n_workers
# (base_runner.py:286) -- shard the work with `env: slurm` if this becomes the
# bottleneck.
#
# NO LOCK. This reads checkpoints and writes ${inference_dir}; it never touches
# exp_dir's checkpoints, so it can run alongside training. It decodes a PINNED
# snapshot rather than valid.acc.ave_10best.pth, which training keeps rewriting.

set -euo pipefail
submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
repo_root=$submit_dir
if [ ! -d "$repo_root/egs3" ]; then
    repo_root=$(git -C "$submit_dir" rev-parse --show-toplevel)
fi
recipe_dir="$repo_root/egs3/must_c/st"
training_config=${TRAINING_CONFIG:-conf/tuning/train_st_conformer.yaml}
inference_config=${INFERENCE_CONFIG:-conf/inference.yaml}
metrics_config=${METRICS_CONFIG:-conf/metrics.yaml}

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export EGS3_HF_CACHE_DIR="$recipe_dir/data/hf"
export MUST_C=/work/hdd/bbjs/shared/corpora/must-c_v1.2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

set +u; . "$repo_root/tools/activate_python.sh"; set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$recipe_dir"

echo "=== MuST-C en-de ST: infer + measure ==="
echo "node      : ${SLURMD_NODENAME:-$(hostname)}  job ${SLURM_JOB_ID:-?}"
echo "started   : $(date -Is)"
echo "inference : $inference_config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python -u run.py --stages infer measure \
    --training_config "$training_config" \
    --inference_config "$inference_config" \
    --metrics_config "$metrics_config"

echo "finished  : $(date -Is)"
