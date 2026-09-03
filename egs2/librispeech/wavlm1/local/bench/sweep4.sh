#!/usr/bin/env bash
# Phase 4: push batch_bins past the fragmentation-induced OOM.
# expandable_segments:True is now exported by the python wrapper on the compute
# node (sbatch --export=PATH blocks caller env), which should recover the
# ~6 GiB that was reserved-but-unallocated when 48M failed.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
ITERS=${SWEEP_ITERS:-1200}
OUT=local/bench/sweep4_results.tsv
[ -f "$OUT" ] || printf "point\tbatch_bins\tstatus\n" > "$OUT"

run_point () {
    local name=$1 bins=$2
    local exp="exp_sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] point ${name}: batch_bins=${bins} (expandable_segments) ==="
    rm -rf "${exp}"
    ./run.sh --stage 7 --stop_stage 7 \
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
    printf "%s\t%s\t%s\n" "${name}" "${bins}" "${status}" >> "${OUT}"
}

run_point es48M 48000000
run_point es64M 64000000
run_point es80M 80000000
echo "=== [$(date '+%F %T')] phase 4 complete ==="
cat "${OUT}"
