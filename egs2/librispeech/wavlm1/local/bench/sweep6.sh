#!/usr/bin/env bash
# Phase 6: does GPU COUNT matter, at fixed per-GPU work?
#
# batch_bins is a GLOBAL budget that abs_task shards across ranks, so per-GPU
# work = batch_bins / ngpu. Two questions, separated:
#
#   A) ngpu=4 @ 18M  -> per-GPU budget 4.5M, IDENTICAL to the 8-GPU 36M point.
#      Same compute per GPU, half the all-reduce participants. If power rises
#      above 486 W, collective overhead was costing us and fewer GPUs helps.
#      If it lands at ~486 W, GPU count is irrelevant to per-GPU power.
#
#   B) ngpu=4 @ 36M  -> per-GPU budget 9.0M, DOUBLE the current point. This is
#      the only way fewer GPUs raises power, and it should OOM: 36M already
#      peaks at 64.2 GB of 79.2 with 4.5M per GPU.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
ITERS=${SWEEP_ITERS:-1000}
OUT=local/bench/sweep6_results.tsv
[ -f "$OUT" ] || printf "point\tngpu\tbatch_bins\tper_gpu_M\tstatus\n" > "$OUT"

run_point () {
    local name=$1 ngpu=$2 bins=$3
    local exp="exp_sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] point ${name}: ngpu=${ngpu} batch_bins=${bins} ==="
    rm -rf "${exp}"
    ./run.sh --stage 7 --stop_stage 7 --ngpu "${ngpu}" \
        --train_start_iter 0 --train_stop_iter 0 --expdir "exp_sweep" \
        --n_clusters "500" --features_km "random" --layers_km "0" --portion_km 1.0 \
        --train_configs "${CFG}" \
        --python "$(pwd)/local/bench/python_with_powerlog.sh" \
        --wavlm_args "--num_iters_per_epoch ${ITERS} --max_epoch 1 --num_att_plot 0 \
                      --keep_nbest_models 1 --patience none \
                      --batch_bins ${bins} --accum_grad 1 --unused_parameters false" \
        > "local/bench/sweep_${name}.log" 2>&1
    local tl status
    tl="${exp}/train.log"
    if   grep -qiE "CUDA out of memory|OutOfMemoryError" "${tl}" 2>/dev/null; then status="OOM"
    elif grep -qiE "Traceback|RuntimeError" "${tl}" 2>/dev/null;            then status="ERROR"
    elif grep -q  "training was finished" "${tl}" 2>/dev/null;              then status="ok"
    else status="incomplete"; fi
    grep -oE "1epoch results: \[train\].*" "${tl}" 2>/dev/null | head -1 > "local/bench/timing_${name}.txt"
    grep -oE "mini-batch sizes summary:.*" "${tl}" 2>/dev/null | head -1 >> "local/bench/timing_${name}.txt"
    grep -oE "gpu_max_cached_mem_GB=[0-9.]+" "${tl}" 2>/dev/null | tail -1 >> "local/bench/timing_${name}.txt"
    echo "    -> ${status}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${ngpu}" "${bins}" \
        "$(awk -v b=${bins} -v n=${ngpu} 'BEGIN{printf "%.2f", b/n/1e6}')" "${status}" >> "${OUT}"
}

run_point g4_18M 4 18000000   # same per-GPU work as 8-GPU 36M
run_point g4_36M 4 36000000   # double per-GPU work; expected OOM
run_point g2_9M  2  9000000   # same per-GPU work again, 2 GPUs
echo "=== [$(date '+%F %T')] phase 6 complete ==="
cat "${OUT}"
