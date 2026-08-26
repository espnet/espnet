#!/usr/bin/env bash
# Chain WavLM iterations 0 -> 1. Iteration 0 skips stage 5 (the MFCC k-means
# pseudo-labels were carried over from hubert1 and are model-independent) and
# starts at stage 6; iteration 1 runs the full 5 -> 7.
set -e
set -u
set -o pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }

log "=== WavLM iteration 0: stages 6-7 (stats + training; k-means reused) ==="
./run.sh --stage 6 --stop_stage 7 --train_start_iter 0 --train_stop_iter 0
log "=== WavLM iteration 0 finished ==="

log "=== WavLM iteration 1: stages 5-7 (k-means on iter-0 layer 6, stats, training) ==="
./run.sh --stage 5 --stop_stage 7 --train_start_iter 1 --train_stop_iter 1
log "=== WavLM iteration 1 finished ==="
