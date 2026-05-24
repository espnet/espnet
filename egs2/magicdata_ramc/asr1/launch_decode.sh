#!/usr/bin/env bash
# Launcher used inside the `asr_decode` tmux session.
# Stays alive (exec bash) after decoding exits so the user can inspect output / logs.
set -u
cd "$(dirname "$0")"
mkdir -p logs
echo "=== Launched at $(date -Is) on host $(hostname) ==="
bash ./run.sh --stage 12 --stop-stage 13 2>&1 | tee -a logs/decode_wrapper.log
status=${PIPESTATUS[0]}
echo "=== run.sh (decode+score) exited with status ${status} at $(date -Is) ==="
exec bash
