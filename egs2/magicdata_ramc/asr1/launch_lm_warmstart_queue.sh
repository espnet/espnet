#!/usr/bin/env bash
# Queue + run the LM-warmstart experiment:
#   (1) Wait until both GPUs drop below 3 GB used.
#   (2) Train a Transformer LM warm-started from
#       espnet/jiyangtang_magicdata_asr_conformer_lm_transformer.
#   (3) Decode the EXISTING e_branchformer_scratch ASR model using this new LM
#       (stages 12-13). The from-scratch ASR + from-scratch LM number is
#       already recorded (dev 13.7 / test 17.6) — the only delta is the LM.
set -u
cd "$(dirname "$0")"
mkdir -p logs

THRESHOLD_MIB=3000

idle() {
    local mems
    mems=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "${mems}" ] && return 1
    while IFS= read -r m; do
        [ "${m}" -ge "${THRESHOLD_MIB}" ] && return 1
    done <<< "${mems}"
    return 0
}

echo "=== Queue waiter started at $(date -Is) on host $(hostname) ==="
echo "=== Waiting for both GPUs to drop below ${THRESHOLD_MIB} MiB ==="
while ! idle; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader | sed 's/^/  /'
    sleep 60
done
echo "=== GPUs idle at $(date -Is) ==="

echo "--- LM warm-start training ---"
bash ./train_lm_warmstart.sh 2>&1 \
    | tee -a logs/train_lm_warmstart_wrapper.log
echo "=== train_lm_warmstart.sh exited with status ${PIPESTATUS[0]} at $(date -Is) ==="

echo "--- Decode + score with warm-start LM (stages 12-13) ---"
bash ./run_decode_warmstart_lm.sh --stage 12 --stop-stage 13 2>&1 \
    | tee -a logs/decode_with_warmstart_lm_wrapper.log
echo "=== run_decode_warmstart_lm.sh exited with status ${PIPESTATUS[0]} at $(date -Is) ==="

echo ""
echo "=== full pipeline finished at $(date -Is) ==="
exec bash
