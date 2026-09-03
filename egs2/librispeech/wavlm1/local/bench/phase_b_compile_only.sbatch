#!/usr/bin/env bash
#SBATCH --job-name=wavlm_compile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=900G
#SBATCH --time=10:00:00
#
# CPU/memory sizing matters: 8 ranks x num_workers=8 = 64 dataloader processes.
# A first attempt requested only 32 CPUs / 150G and ran ~3.5x slower
# (forward_time 1.85 s vs ~0.5 s) at 229.8 W instead of the 494.0 W measured for
# the same batch_bins via run.sh -- which submits with cpu=192, mem=900G. The
# benchmark is meaningless if the CPU allocation differs from the runs it is
# being compared against.
#SBATCH --output=/mnt/weka/mark/espnet/.runlogs/phase_b_compile_%j.log
#
# Phase B as a SELF-CONTAINED slurm job.
#
# Previous attempts drove training from a login-node shell via
# run.sh -> wavlm.sh -> launch.py -> slurm.pl. That chain dies whenever the
# login session is signalled or the node is switched, which has now killed the
# benchmark twice (12:55 today at iteration 2500/10000). Here the whole thing
# runs INSIDE one allocation: nothing on a login node can interrupt it, and both
# comparison points share a single queue wait.
set -u
cd /mnt/weka/mark/espnet/egs2/librispeech/wavlm1
# Source the recipe's path.sh, not just the python env. It exports
# OMP_NUM_THREADS=1, LC_ALL, PYTHONIOENCODING and NCCL_SOCKET_IFNAME. Skipping it
# left OMP_NUM_THREADS unset, so each of the 8 ranks spawned up to 192 OpenMP
# threads on a 208-CPU node: forward_time 0.967 s vs 0.170 s in the run.sh
# reference, while the GPU-bound backward pass was unchanged (0.224 vs 0.221).
. ./path.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BINS=36000000          # highest measured power for iteration 0: 494.0 W
ITERS=10000
RES=local/bench/phase_b_results.tsv
[ -f "$RES" ] || printf "point\tbatch_bins\titers\tcompile\tstatus\telapsed_s\n" > "$RES"

run_one () {
    local name=$1 compile=$2
    local out="exp_phaseb/${name}"
    local pdir="local/bench/power_phaseb/${name}"
    rm -rf "$out"; mkdir -p "$out" "$pdir"

    nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm \
               --format=csv,nounits -l 5 > "${pdir}/power.csv" 2>/dev/null &
    local sampler=$!

    echo "=== [$(date '+%F %T')] ${name}: batch_bins=${BINS} iters=${ITERS} compile=${compile} ==="
    local t0=$(date +%s)
    python3 -m espnet2.bin.wavlm_train \
        --use_preprocessor true --normalize null --token_type word \
        --token_list data/en_token_list_kmeans_iter0_rand100_100clusters/word/tokens.txt \
        --num_classes 100 \
        --train_data_path_and_name_and_type dump/raw/train_960/wav.scp,speech,sound \
        --train_data_path_and_name_and_type dump/raw/train_960/text.km.kmeans_iter0_rand100_train_960_portion1.0,text,text \
        --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
        --valid_data_path_and_name_and_type dump/raw/dev/text.km.kmeans_iter0_rand100_train_960_portion1.0,text,text \
        --train_shape_file exp_it0rand/wavlm_iter0_stats_raw/train/speech_shape \
        --train_shape_file exp_it0rand/wavlm_iter0_stats_raw/train/text_shape.word \
        --valid_shape_file exp_it0rand/wavlm_iter0_stats_raw/valid/speech_shape \
        --valid_shape_file exp_it0rand/wavlm_iter0_stats_raw/valid/text_shape.word \
        --config conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0rand.yaml \
        --frontend_conf fs=16k --resume false \
        --fold_length 80000 --fold_length 400 \
        --output_dir "$out" \
        --num_iters_per_epoch ${ITERS} --max_epoch 1 --num_att_plot 0 \
        --keep_nbest_models 1 --patience none \
        --batch_bins ${BINS} --accum_grad 1 --unused_parameters false \
        --use_torch_compile ${compile} \
        --ngpu 8 --multiprocessing_distributed true \
        > "local/bench/phaseb_${name}.log" 2>&1
    local t1=$(date +%s)
    kill ${sampler} 2>/dev/null

    local status
    local LOG="local/bench/phaseb_${name}.log"
    if   grep -qiE "CUDA out of memory" "$LOG" 2>/dev/null;     then status=OOM
    elif grep -q "training was finished" "$LOG" 2>/dev/null;    then status=ok
    elif grep -qiE "Traceback|RuntimeError" "$LOG" 2>/dev/null; then status=ERROR
    else status=incomplete; fi
    echo "    -> ${status} in $((t1-t0)) s"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$BINS" "$ITERS" "$compile" "$status" "$((t1-t0))" >> "$RES"
}


run_one B_compile   true
echo "=== [$(date '+%F %T')] PHASE B COMPLETE ==="
cat "$RES"
