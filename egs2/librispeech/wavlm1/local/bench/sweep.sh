#!/usr/bin/env bash
# Sweep training knobs and record GPU power for each setting.
# Each point runs a short training burst (power reaches steady state in ~2 min)
# and is analysed separately. OOM or other failures are recorded, not fatal.
set -u
cd "$(dirname "$0")/../.."

CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml

# Every point reuses one collect-stats output (shapes do not depend on batching).
if [ ! -f exp_sweep/wavlm_iter0_stats_raw/train/speech_shape ]; then
    echo "ERROR: exp_sweep/wavlm_iter0_stats_raw/train/speech_shape missing." >&2
    echo "       ln -sfn ../exp_bench/wavlm_iter0_stats_raw exp_sweep/wavlm_iter0_stats_raw" >&2
    exit 1
fi
ITERS=${SWEEP_ITERS:-1200}
OUT=local/bench/sweep_results.tsv
[ -f "$OUT" ] || printf "point\tbatch_bins\taccum_grad\tnum_workers\tstatus\tpower_csv\n" > "$OUT"

run_point () {
    local name=$1 bins=$2 accum=$3 workers=$4
    # wavlm.sh derives the experiment dir from the config name, not from $name,
    # so clear that specific dir (never the whole expdir -- it holds the shared
    # stats symlink that every point reuses).
    local exp="exp_sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] point ${name}: batch_bins=${bins} accum_grad=${accum} num_workers=${workers} ==="
    rm -rf "${exp}"
    export BENCH_POWER_DIR="$(pwd)/local/bench/power_sweep/${name}"
    export BENCH_POWER_INTERVAL=2
    rm -rf "${BENCH_POWER_DIR}"; mkdir -p "${BENCH_POWER_DIR}"

    ./run.sh --stage 7 --stop_stage 7 \
        --train_start_iter 0 --train_stop_iter 0 \
        --expdir "exp_sweep" \
        --n_clusters "500" --features_km "random" --layers_km "0" --portion_km 1.0 \
        --train_configs "${CFG}" \
        --python "$(pwd)/local/bench/python_with_powerlog.sh" \
        --wavlm_args "--num_iters_per_epoch ${ITERS} --max_epoch 1 --num_att_plot 0 \
                      --keep_nbest_models 1 --patience none \
                      --batch_bins ${bins} --accum_grad ${accum} --num_workers ${workers}" \
        > "local/bench/sweep_${name}.log" 2>&1

    local tl status csv
    tl=$(ls -t exp_sweep/*/train.log 2>/dev/null | head -1)
    csv=$(ls -t "${BENCH_POWER_DIR}"/*.csv 2>/dev/null | head -1)
    if [ -z "${tl}" ]; then
        status="no-train-log"
    elif grep -qiE "CUDA out of memory|torch.OutOfMemoryError" "${tl}"; then
        status="OOM"
    elif grep -qiE "Traceback|RuntimeError" "${tl}"; then
        status="ERROR"
    elif grep -q "training was finished" "${tl}"; then
        status="ok"
    else
        status="incomplete"
    fi
    echo "    -> ${status}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${name}" "${bins}" "${accum}" "${workers}" "${status}" "${csv}" >> "${OUT}"
}

# batch_bins ladder first (the lever the user asked to max out), fixed accum/workers
run_point bins6M   6000000  4 8
run_point bins8M   8000000  4 8
run_point bins12M 12000000  4 8
run_point bins16M 16000000  4 8
echo "=== [$(date '+%F %T')] batch_bins ladder complete ==="
cat "${OUT}"
