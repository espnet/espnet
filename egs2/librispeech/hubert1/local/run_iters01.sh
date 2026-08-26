#!/usr/bin/env bash
# Chain HuBERT iterations 0 -> 1. Iteration 0 skips stage 5 (the MFCC k-means and
# its pseudo-labels already exist from the previous run) and stage 6 (the stats
# dir already exists); iteration 1 runs the full 5 -> 7.
# NOTE: deliberately NOT using `set -e`. espnet's trainer exits non-zero even on
# a clean finish, because NCCL's process-group teardown aborts after the training
# loop (destroy_process_group() is never called). A `set -e` guard here therefore
# reads a successful run as a failure -- which is exactly what silently stopped
# this driver from chaining into iteration 1. Success is judged by inspecting the
# artifacts instead of by the exit status.
set -u
set -o pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }
MODEL=hubert
FEATS=raw

# Verify iteration ${iter} produced a usable model rather than trusting $?.
check_iter() {
    local iter=$1 cfg=$2
    local exp="exp/${MODEL}_iter${iter}_$(basename "${cfg}" .yaml)_${FEATS}"
    local ok=true
    for f in "${exp}/valid.loss.best.pth" "${exp}/config.yaml"; do
        if [ -e "$f" ]; then log "  present: $f"; else log "  MISSING: $f"; ok=false; fi
    done
    if grep -q "Early stopping\|The training was finished at" "${exp}/train.log" 2>/dev/null; then
        log "  iteration ${iter} reached a normal stopping point"
    else
        log "  WARNING: no normal stop marker in ${exp}/train.log"
        ok=false
    fi
    $ok
}


log "=== HuBERT iteration 0: stage 7 (training only; k-means + stats reused) ==="
./run.sh --stage 7 --stop_stage 7 --train_start_iter 0 --train_stop_iter 0
if check_iter 0 "conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it0.yaml"; then
    log "=== iteration 0 finished successfully ==="
else
    log "=== iteration 0 did NOT complete cleanly; not starting iteration 1 ==="
    exit 1
fi

log "=== HuBERT iteration 1: stages 5-7 (k-means on iter-0 layer 6, stats, training) ==="
./run.sh --stage 5 --stop_stage 7 --train_start_iter 1 --train_stop_iter 1
if check_iter 1 "conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it1.yaml"; then
    log "=== iteration 1 finished successfully ==="
else
    log "=== iteration 1 did NOT complete cleanly ==="
fi
