#!/bin/bash
#SBATCH --job-name=must_c_st_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --qos=bbjs-delta-gpu
#SBATCH --output=/work/hdd/bbjs/sjin2/espnet/training_log/must_c_st_train_%j_outer.log

# MuST-C en-de speech translation, egs2 target (STTask, two BPE vocabularies).
#
#   sbatch egs3/must_c/st/submit_train.sh                  # first or next link
#   ./chain_submit.sh 3                                    # queue 3 links
#
# TWO PARTITIONS. gpuA100x4 alone estimated a 25 h queue wait (measured
# 2026-09-18: Slurm's StartTime for job 22184960 was a full day out), so
# gpuA40x4 is listed too and Slurm takes whichever frees first. The A40 has
# 48 GB against the A100's 40 GB, so the measured ~29-30 GiB peak fits with
# more room, not less; it is the slower card, which the 3-link chain absorbs.
#
# ONE GPU, matching egs2: st.sh's default is ngpu=1 and egs2/must_c/st1/run.sh
# does not override it. `num_device: 1` in the config and --gpus-per-node=1
# here must stay in step; batch_bins is PER GPU in espnet3, so adding GPUs
# without rescaling batch_bins/warmup_steps changes the effective batch.
# A single rank also means no srun re-exec is needed (see commonvoice's
# submit_train.sh for the multi-rank form and why it must use srun).
#
# RESUME. espnet3 runs `trainer.fit(**config.fit)` and the base config leaves
# `fit: {}`, i.e. from scratch. last.ckpt is refreshed at the end of every
# training epoch by the default callbacks, so a link that finds one switches to
# the _resume config; the first link, with no checkpoint, uses the base config.
# Override with TRAINING_CONFIG=... to force either.

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
repo_root=$submit_dir
if [ ! -d "$repo_root/egs3" ]; then
    repo_root=$(git -C "$submit_dir" rev-parse --show-toplevel)
fi
recipe_dir="$repo_root/egs3/must_c/st"
exp_dir="$recipe_dir/exp/train_st_conformer"

if [ -n "${TRAINING_CONFIG:-}" ]; then
    training_config=$TRAINING_CONFIG
elif [ -e "$exp_dir/last.ckpt" ]; then
    training_config=conf/tuning/train_st_conformer_resume.yaml
else
    training_config=conf/tuning/train_st_conformer.yaml
fi

mkdir -p "$repo_root/training_log"
exec >"$repo_root/training_log/must_c_st_train_${SLURM_JOB_ID:-nojob}.log" 2>&1

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export EGS3_HF_CACHE_DIR="$recipe_dir/data/hf"
export MUST_C=/work/hdd/bbjs/shared/corpora/must-c_v1.2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B: use the key when present, but only online if this node can actually
# reach the API -- an unreachable endpoint stalls the run on upload retries.
wandb_key_file="$repo_root/.wandb.key"
if [ -s "$wandb_key_file" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$wandb_key_file")"
    if curl -sS -o /dev/null -m 15 https://api.wandb.ai 2>/dev/null; then
        export WANDB_MODE=online
    else
        export WANDB_MODE=offline
    fi
else
    export WANDB_MODE=offline
fi
export WANDB_DIR="$exp_dir"

set +u
. "$repo_root/tools/activate_python.sh"
set -u

# espnet2.tasks.st pulls in libicui18n, which needs CXXABI_1.3.15; the system
# /lib64/libstdc++.so.6 is older, so the env's own copy must come first.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$recipe_dir"

# MUTUAL EXCLUSION with chain_interactive.sh. Both write this exp_dir, and two
# concurrent trainers would interleave writes to last.ckpt -- the loser's steps
# are then silently discarded on the next resume. A batch allocation is the
# scarcer resource, so this one WAITS up to 20 min for the lock rather than
# yielding immediately; an interactive session is cheap to lose.
. "$recipe_dir/train_lock.sh"
lockdir="$exp_dir/.training.lock"
# 70 min, longer than one interactive session's 1 h wall. An interactive
# session holds the lock for at most its hour, so a batch link that starts
# mid-session waits it out and then takes over for the full 48 h, instead of
# giving up after 20 min and burning a chain link (which is what job 22187285
# did at 10:41). The idle GPU time is a one-off handover cost.
if ! acquire_lock "$lockdir" "${SLURM_JOB_ID:-nojob}" 4200; then
    echo "another trainer holds the lock and did not release within 70 min; exiting"
    exit 0
fi
trap 'release_lock "$lockdir" "${SLURM_JOB_ID:-nojob}"' EXIT

echo "=== MuST-C en-de ST training (Delta A100/A40, 1 GPU) ==="
echo "job     : ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname)}"
echo "started : $(date -Is)"
echo "config  : $training_config"
echo "wandb   : ${WANDB_MODE}"
echo "python  : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# NOTE: egs2 st.sh stage 4 ("Remove long/short data", 0.1 s .. 20 s on train
# and valid) is applied by the Dataset itself now -- see `filter:` in
# dataset/config.yaml. It therefore also shapes collect_stats, so feats_shape
# and feats_stats.npz both already describe the trimmed set and there is
# nothing to post-process here.

# Fail fast rather than 40 minutes in: without the text shape files the batcher
# silently ignores both text streams and OOMs on a short-utterance batch.
for f in exp/stats/train/text_shape.bpe exp/stats/train/src_text_shape.bpe \
         exp/stats/valid/text_shape.bpe exp/stats/valid/src_text_shape.bpe; do
    if [ ! -s "$f" ]; then
        echo "missing $f -- run gen_text_shape.py (see the batch_bins note in the config)" >&2
        exit 1
    fi
done
echo

python -u run.py --stages train --training_config "$training_config"

echo "finished: $(date -Is)"
