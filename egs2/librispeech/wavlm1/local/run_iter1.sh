#!/usr/bin/env bash
# WavLM iteration 1: stage 5 (k-means on iter-0 layer 6) -> 6 (stats) -> 7 (train).
#
# NOTE: deliberately NOT using `set -e` around run.sh. espnet's trainer exits
# non-zero even on a clean finish, because NCCL's process-group teardown aborts
# after the training loop (destroy_process_group() is never called). A `set -e`
# guard here therefore reports a false failure -- and in the previous chained
# driver it aborted the run before iteration 1 could start. Success is judged by
# inspecting the artifacts instead of by the exit status.
set -u
set -o pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }
EXP=exp/wavlm_iter1_train_ssl_torchaudiowavlm_base_960h_pretrain_it1_raw

log "=== WavLM iteration 1: stages 5-7 (k-means on iter-0 layer 6, stats, training) ==="
./run.sh --stage 5 --stop_stage 7 --train_start_iter 1 --train_stop_iter 1
status=$?
log "run.sh exited with status ${status} (non-zero is expected on a clean finish; verifying artifacts)"

ok=true
for f in "${EXP}/valid.loss.best.pth" "${EXP}/config.yaml"; do
    if [ -e "$f" ]; then log "  present: $f"; else log "  MISSING: $f"; ok=false; fi
done
if grep -q "Early stopping\|The training was finished at" "${EXP}/train.log" 2>/dev/null; then
    log "  training reached a normal stopping point"
else
    log "  WARNING: no normal stop marker found in ${EXP}/train.log"
    ok=false
fi

if $ok; then
    log "=== WavLM iteration 1 finished successfully ==="
else
    log "=== WavLM iteration 1 did NOT complete cleanly - inspect ${EXP}/train.log ==="
fi
