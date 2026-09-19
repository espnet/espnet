#!/bin/bash
#SBATCH --job-name=must_c_st_collect_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu-preempt,cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --requeue

# MuST-C en-de ST: train the two SentencePiece models (a no-op once they
# exist -- STSystem skips a side whose bpe.model and tokens.txt are already
# there) and then collect feature statistics.
#
#   sbatch egs3/must_c/st/submit_collect_stats.sh
#
# Self-contained: the `parallel:` block in the training config is `env: local`,
# so the 16 dask workers run INSIDE this allocation rather than as 16 more
# SLURM jobs. That is why this asks for 16 cores rather than being a thin
# driver -- see the long note on `parallel:` in the config for why the
# distributed shape was abandoned.
#
# 64 GB for 16 worker processes: each builds its own copy of the 57.6M-param
# model plus a torch runtime, roughly 2 GB apiece.
#
# TWO PARTITIONS, and --requeue. Slurm starts the job in whichever of
# cpu-preempt / cpu can run it first. cpu-preempt is PriorityTier=30 against
# cpu's 100, so it picks up nodes that would otherwise idle, at the price of
# being preempted when a higher-tier job wants them; PreemptMode=REQUEUE with
# GraceTime=300 then puts this job back in the queue rather than killing it.
#
# That is safe HERE specifically because the stage is resumable: espnet3's
# BaseRunner defaults to resume=True and skips any shard carrying a done
# marker, so a requeued run redoes only the shards that were in flight. The
# stale-lock sweep below is what makes that work -- a preempted worker leaves
# its lock behind, and without the sweep the requeued run would abort with
# "Shard is already locked by another runner".

set -euo pipefail
submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
repo_root=$submit_dir
if [ ! -d "$repo_root/egs3" ]; then
    repo_root=$(git -C "$submit_dir" rev-parse --show-toplevel)
fi
recipe_dir="$repo_root/egs3/must_c/st"
config=conf/tuning/train_st_conformer.yaml

mkdir -p "$repo_root/data_prep_log"
exec >"$repo_root/data_prep_log/must_c_st_collect_stats.log" 2>&1

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export EGS3_HF_CACHE_DIR="$recipe_dir/data/hf"
export MUST_C=/work/hdd/bbjs/shared/corpora/must-c_v1.2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

set +u
. "$repo_root/tools/activate_python.sh"
set -u

# espnet2.tasks.st pulls in libicui18n, which needs CXXABI_1.3.15; the system
# /lib64/libstdc++.so.6 is older, so the env's own copy must come first or the
# task class fails to import.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$recipe_dir"

# Clear stale shard locks from an interrupted run. espnet3's BaseRunner takes
# a `lock` file per shard and removes it on completion, so a cancelled job
# leaves locks behind with no results; the next run then dies on
#   RuntimeError: Shard is already locked by another runner: exp/stats/train/split.N
# A lock is only ever stale here because this recipe runs one driver at a time,
# and a shard that really finished is identified by its done marker, not the
# lock. Shards with results are left completely alone so `resume` still works.
for shard in exp/stats/*/split.*; do
    [ -d "$shard" ] || continue
    if [ -f "$shard/lock" ] && [ "$(find "$shard" -type f ! -name lock | wc -l)" -eq 0 ]; then
        echo "clearing stale lock in $shard"
        rm -f "$shard/lock"
    fi
done

echo "=== MuST-C en-de ST train_tokenizer + collect_stats ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "config  : $config"
echo "python  : $(which python)"

python -u run.py --stages train_tokenizer collect_stats --training_config "$config"

# The batcher needs text_shape.bpe / src_text_shape.bpe as well as
# feats_shape -- espnet3's collect_stats does not emit them and egs2 passes all
# three. Without them a short-utterance batch grows unbounded and OOMs; see the
# batch_bins note in the training config.
echo "=== generating text shape files ==="
python -u gen_text_shape.py

echo "finished: $(date -Is)"
ls -la exp/stats/train exp/stats/valid 2>/dev/null || true
