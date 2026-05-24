#!/usr/bin/env bash
# Launcher used inside the `asr_train_sp_lm` tmux session.
# Resumes from stage 10 (ASR stats + train) and then runs decode+score.
# Stages 2-7 already done in the previous attempt — SP data, dumps, token list,
# LM stats, and LM training (15 epochs) all completed and are reused.
#
# NOTE: each asr.sh invocation is run unconditionally (no `set -e` between them)
# so the harmless NCCL teardown exit-code-1 at the end of a DDP training run does
# not block the subsequent decode stage.
set -u
cd "$(dirname "$0")"
mkdir -p logs
echo "=== Launched at $(date -Is) on host $(hostname) ==="

echo "--- Stage 10-11: ASR stats + training ---"
bash ./run_branchformer_sp_lm.sh --stage 10 --stop-stage 11 2>&1 | tee -a logs/train_sp_lm_wrapper.log
status_train=${PIPESTATUS[0]}
echo "=== stages 10-11 exited with status ${status_train} at $(date -Is) ==="

echo "--- Stage 12-13: decode + score (with LM rescoring) ---"
bash ./run_branchformer_sp_lm.sh --stage 12 --stop-stage 13 2>&1 | tee -a logs/decode_sp_lm_wrapper.log
status_decode=${PIPESTATUS[0]}
echo "=== stages 12-13 exited with status ${status_decode} at $(date -Is) ==="

exec bash
