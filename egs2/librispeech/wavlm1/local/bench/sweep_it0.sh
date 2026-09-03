#!/usr/bin/env bash
# Iteration-0 profile (100 clusters, label_downsampling 2), WavLM large.
#
# Phase A: find the batch_bins that maximises per-GPU power before OOM.
# Phase B: at that setting, run the full 10,000 steps twice -- with and without
#          torch.compile -- and compare wall time as well as power.
#
# NOTE on scaling: an iteration-0 utterance costs 0.51x the batch_bins of an
# iteration-2 one (label term is 24% of budget vs 61%), so the same batch_bins
# buys ~1.95x more utterances -- and therefore ~1.95x the transformer frames,
# which is what actually drives memory. The OOM point is expected LOWER in
# absolute batch_bins than iteration 2's 36M, not higher.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0rand.yaml
OUT=local/bench/sweep_it0_results.tsv
[ -f "$OUT" ] || printf "point\tbatch_bins\titers\tcompile\tstatus\telapsed_s\n" > "$OUT"

# wait for the shared stats
while [ ! -f exp_it0rand/wavlm_iter0_stats_raw/train/speech_shape ]; do sleep 30; done
ln -sfn ../exp_it0rand/wavlm_iter0_stats_raw exp_it0sweep/wavlm_iter0_stats_raw 2>/dev/null || {
    mkdir -p exp_it0sweep
    ln -sfn ../exp_it0rand/wavlm_iter0_stats_raw exp_it0sweep/wavlm_iter0_stats_raw; }

run_point () {
    local name=$1 bins=$2 iters=$3 compile=$4
    local exp="exp_it0sweep/wavlm_iter0_$(basename "${CFG}" .yaml)_raw"
    echo "=== [$(date '+%F %T')] ${name}: bins=${bins} iters=${iters} compile=${compile} ==="
    rm -rf "${exp}"
    local t0=$(date +%s)
    ./run.sh --stage 7 --stop_stage 7 \
        --train_start_iter 0 --train_stop_iter 0 --expdir "exp_it0sweep" \
        --n_clusters "100" --features_km "rand100" --layers_km "0" --portion_km 1.0 \
        --train_configs "${CFG}" \
        --python "$(pwd)/local/bench/python_with_powerlog.sh" \
        --wavlm_args "--num_iters_per_epoch ${iters} --max_epoch 1 --num_att_plot 0 \
                      --keep_nbest_models 1 --patience none \
                      --batch_bins ${bins} --accum_grad 1 --unused_parameters false \
                      --use_torch_compile ${compile}" \
        > "local/bench/sweep_${name}.log" 2>&1
    local t1=$(date +%s) tl status
    tl="${exp}/train.log"
    if   grep -qiE "CUDA out of memory|OutOfMemoryError" "${tl}" 2>/dev/null; then status="OOM"
    elif grep -qiE "Traceback|RuntimeError" "${tl}" 2>/dev/null;            then status="ERROR"
    elif grep -q  "training was finished" "${tl}" 2>/dev/null;              then status="ok"
    else status="incomplete"; fi
    grep -oE "1epoch results: \[train\].*" "${tl}" 2>/dev/null | head -1 > "local/bench/timing_${name}.txt"
    grep -oE "mini-batch sizes summary:.*" "${tl}" 2>/dev/null | head -1 >> "local/bench/timing_${name}.txt"
    grep -oE "gpu_max_cached_mem_GB=[0-9.]+" "${tl}" 2>/dev/null | tail -1 >> "local/bench/timing_${name}.txt"
    echo "    -> ${status} in $((t1-t0)) s"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${name}" "${bins}" "${iters}" "${compile}" "${status}" "$((t1-t0))" >> "${OUT}"
}

# Phase A: short points to locate the power maximum / OOM edge
run_point i0_12M 12000000 1200 false
run_point i0_18M 18000000 1200 false
run_point i0_24M 24000000 1200 false
run_point i0_30M 30000000 1200 false
echo "=== [$(date '+%F %T')] iteration-0 batch_bins ladder complete ==="
cat "${OUT}"
