#!/usr/bin/env bash
# Runs two E-Branchformer comparison experiments back-to-back:
#   1) from scratch       (asr_tag: e_branchformer_scratch)
#   2) warm-start AISHELL (asr_tag: e_branchformer_warmstart_aishell)
#
# For each, we invoke training (stage 11) and decode+score (stage 12-13)
# as SEPARATE asr.sh calls so that NCCL-teardown exit-code-1 noise at the
# end of training does not block the subsequent decode stage.
# Stages 2-7 (data prep + LM) and stage 10 (ASR stats) are already done.
set -u
cd "$(dirname "$0")"
mkdir -p logs
echo "=== Launched at $(date -Is) on host $(hostname) ==="

run_pair() {
    local tag="$1"; shift
    local warmstart="$1"; shift

    echo ""
    echo "############################################################"
    echo "# Experiment: ${tag}"
    echo "# Started at $(date -Is)"
    echo "############################################################"

    echo "--- Stage 11: training (${tag}) ---"
    WARMSTART=${warmstart} bash ./run_e_branchformer.sh --stage 11 --stop-stage 11 2>&1 \
        | tee -a "logs/train_${tag}_wrapper.log"
    echo "=== training (${tag}) exited with status ${PIPESTATUS[0]} at $(date -Is) ==="

    echo "--- Stage 12-13: decode + score (${tag}) ---"
    WARMSTART=${warmstart} bash ./run_e_branchformer.sh --stage 12 --stop-stage 13 2>&1 \
        | tee -a "logs/decode_${tag}_wrapper.log"
    echo "=== decode (${tag}) exited with status ${PIPESTATUS[0]} at $(date -Is) ==="
}

run_pair "e_branchformer_scratch"           "0"
run_pair "e_branchformer_warmstart_aishell" "1"

echo ""
echo "=== all experiments finished at $(date -Is) ==="
exec bash
