#!/usr/bin/env bash
# Chain WavLM iterations 0 -> 1. Iteration 0 skips stage 5 (the MFCC k-means
# pseudo-labels were carried over from hubert1 and are model-independent) and
# starts at stage 6; iteration 1 runs the full 5 -> 7.
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
MODEL=wavlm
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


log "=== WavLM iteration 0: stages 6-7 (stats + training; k-means reused) ==="
./run.sh --stage 6 --stop_stage 7 --train_start_iter 0 --train_stop_iter 0
if check_iter 0 "conf/tuning/train_ssl_torchaudiowavlm_base_960h_pretrain_it0.yaml"; then
    log "=== WavLM iteration 0 finished successfully ==="
else
    log "=== WavLM iteration 0 did NOT complete cleanly; not starting iteration 1 ==="
    exit 1
fi

log "=== WavLM iteration 1: stages 5-7 (k-means on iter-0 layer 6, stats, training) ==="
./run.sh --stage 5 --stop_stage 7 --train_start_iter 1 --train_stop_iter 1
if check_iter 1 "conf/tuning/train_ssl_torchaudiowavlm_base_960h_pretrain_it1.yaml"; then
    log "=== WavLM iteration 1 finished successfully ==="
else
    log "=== WavLM iteration 1 did NOT complete cleanly ==="
fi
