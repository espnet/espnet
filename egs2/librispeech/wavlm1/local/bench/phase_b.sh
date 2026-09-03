#!/usr/bin/env bash
# Confirm the iteration-0 plateau at 36M, then run Phase B:
# 10,000 steps at the best setting, WITHOUT and WITH torch.compile.
set -u
cd "$(dirname "$0")/../.."
CFG=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0rand.yaml
OUT=local/bench/sweep_it0_results.tsv

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

# A: confirm the plateau (24M and 30M both gave 469.1 W)
run_point i0_36M 36000000 1200 false
echo "=== [$(date '+%F %T')] plateau check done ==="

# B: the 10,000-step comparison the user asked for, at the saturation point
run_point B_nocompile 24000000 10000 false
run_point B_compile   24000000 10000 true
echo "=== [$(date '+%F %T')] PHASE B COMPLETE ==="
cat "${OUT}"
