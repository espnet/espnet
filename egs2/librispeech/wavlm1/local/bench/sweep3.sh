#!/usr/bin/env bash
# Phase 3: find_unused_parameters.
#
# WHY: espnet never calls DDP's no_sync(), so the all-reduce happens on EVERY
# micro-batch regardless of accum_grad -- which is exactly why accum_grad 4 -> 2
# left power unchanged (418.2 -> 418.7 W). What DOES run every backward is
# find_unused_parameters=True (config `unused_parameters: true`, inherited from
# the HuBERT recipe): it walks the whole autograd graph to detect unused params.
# That is the prime suspect for the ~78% of each step that is neither forward,
# backward, optimizer, nor data wait.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
ITERS=${SWEEP_ITERS:-1200}
OUT=local/bench/sweep3_results.tsv
[ -f "$OUT" ] || printf "point\tbatch_bins\taccum_grad\tunused_params\tstatus\n" > "$OUT"

run_point () {
    local name=$1 bins=$2 accum=$3 unused=$4
    local exp="exp_sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] point ${name}: bins=${bins} accum=${accum} unused_parameters=${unused} ==="
    rm -rf "${exp}"
    ./run.sh --stage 7 --stop_stage 7 \
        --train_start_iter 0 --train_stop_iter 0 --expdir "exp_sweep" \
        --n_clusters "500" --features_km "random" --layers_km "0" --portion_km 1.0 \
        --train_configs "${CFG}" \
        --python "$(pwd)/local/bench/python_with_powerlog.sh" \
        --wavlm_args "--num_iters_per_epoch ${ITERS} --max_epoch 1 --num_att_plot 0 \
                      --keep_nbest_models 1 --patience none \
                      --batch_bins ${bins} --accum_grad ${accum} --unused_parameters ${unused}" \
        > "local/bench/sweep_${name}.log" 2>&1
    local tl status
    tl="${exp}/train.log"
    if   grep -qiE "CUDA out of memory|OutOfMemoryError" "${tl}" 2>/dev/null; then status="OOM"
    elif grep -qiE "Traceback|RuntimeError" "${tl}" 2>/dev/null;            then status="ERROR"
    elif grep -q  "training was finished" "${tl}" 2>/dev/null;              then status="ok"
    else status="incomplete"; fi
    grep -oE "1epoch results: \[train\].*" "${tl}" 2>/dev/null | head -1 \
        > "local/bench/timing_${name}.txt"
    echo "    -> ${status}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${bins}" "${accum}" "${unused}" "${status}" >> "${OUT}"
}

run_point up0_16M 16000000 4 false
run_point up0_32M 32000000 1 false
run_point up0_48M 48000000 1 false
echo "=== [$(date '+%F %T')] phase 3 complete ==="
cat "${OUT}"
