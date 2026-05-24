#!/usr/bin/env bash
# Launcher used inside the `asr_train_bf` tmux session.
# Stays alive (exec bash) after training exits so the user can
# inspect output / logs interactively.
set -u
cd "$(dirname "$0")"
mkdir -p logs
echo "=== Launched at $(date -Is) on host $(hostname) ==="
bash ./run_branchformer.sh --stage 11 --stop-stage 11 2>&1 | tee -a logs/train_branchformer_wrapper.log
status=${PIPESTATUS[0]}
echo "=== run_branchformer.sh exited with status ${status} at $(date -Is) ==="
exec bash
