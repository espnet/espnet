#!/usr/bin/env bash
# Phase 2: accum_grad (and a combined point), at the best surviving batch_bins.
# The iteration timing showed forward+backward+optim = 0.130 s of a 0.591 s
# train_time, i.e. ~78% of each step is DDP gradient sync. accum_grad>1 pays
# that all-reduce once per MICRO-batch, so reducing it should attack the
# dominant cost directly.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
ITERS=${SWEEP_ITERS:-1200}
OUT=local/bench/sweep2_results.tsv
[ -f "$OUT" ] || printf "point\tbatch_bins\taccum_grad\tstatus\n" > "$OUT"

run_point () {
    local name=$1 bins=$2 accum=$3
    local exp="exp_sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] point ${name}: batch_bins=${bins} accum_grad=${accum} ==="
    rm -rf "${exp}"
    ./run.sh --stage 7 --stop_stage 7 \
        --train_start_iter 0 --train_stop_iter 0 --expdir "exp_sweep" \
        --n_clusters "500" --features_km "random" --layers_km "0" --portion_km 1.0 \
        --train_configs "${CFG}" \
        --python "$(pwd)/local/bench/python_with_powerlog.sh" \
        --wavlm_args "--num_iters_per_epoch ${ITERS} --max_epoch 1 --num_att_plot 0 \
                      --keep_nbest_models 1 --patience none \
                      --batch_bins ${bins} --accum_grad ${accum}" \
        > "local/bench/sweep_${name}.log" 2>&1
    local tl status
    tl="${exp}/train.log"
    if   grep -qiE "CUDA out of memory|OutOfMemoryError" "${tl}" 2>/dev/null; then status="OOM"
    elif grep -qiE "Traceback|RuntimeError" "${tl}" 2>/dev/null;            then status="ERROR"
    elif grep -q  "training was finished" "${tl}" 2>/dev/null;              then status="ok"
    else status="incomplete"; fi
    echo "    -> ${status}"
    printf "%s\t%s\t%s\t%s\n" "${name}" "${bins}" "${accum}" "${status}" >> "${OUT}"
}

run_point ag2_16M  16000000 2
run_point ag1_16M  16000000 1
run_point ag1_24M  24000000 1
run_point ag1_32M  32000000 1
echo "=== [$(date '+%F %T')] accum_grad sweep complete ==="
cat "${OUT}"
