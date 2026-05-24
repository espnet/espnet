#!/usr/bin/env bash
# Queue launcher: polls until ALL GPUs are idle (<3 GB used) before launching
# the no-LM decode-only run. Polls every 60 s.
#
# The actual decode workload runs on CPU (`--ngpu 0` for inference workers),
# but we still honour the GPU-idle gate so this job doesn't compete with any
# heavier training on the box.
set -u
cd "$(dirname "$0")"
mkdir -p logs

THRESHOLD_MIB=3000   # consider a GPU "idle" if it has < this much memory used

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
echo "=== GPUs idle at $(date -Is) — launching decode ==="

bash ./run_e_branchformer_nolm.sh --stage 12 --stop-stage 13 2>&1 \
    | tee -a logs/decode_e_branchformer_scratch_nolm_wrapper.log
status=${PIPESTATUS[0]}
echo "=== run_e_branchformer_nolm.sh exited with status ${status} at $(date -Is) ==="
exec bash
