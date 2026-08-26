#!/usr/bin/env bash
# Chain HuBERT iterations 0 -> 1. Iteration 0 skips stage 5 (the MFCC k-means and
# its pseudo-labels already exist from the previous run) and stage 6 (the stats
# dir already exists); iteration 1 runs the full 5 -> 7.
set -e
set -u
set -o pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }

log "=== HuBERT iteration 0: stage 7 (training only; k-means + stats reused) ==="
./run.sh --stage 7 --stop_stage 7 --train_start_iter 0 --train_stop_iter 0
log "=== HuBERT iteration 0 finished ==="

log "=== HuBERT iteration 1: stages 5-7 (k-means on iter-0 layer 6, stats, training) ==="
./run.sh --stage 5 --stop_stage 7 --train_start_iter 1 --train_stop_iter 1
log "=== HuBERT iteration 1 finished ==="
