#!/usr/bin/env bash
# Drop-in replacement for `python3` that starts an nvidia-smi power sampler on the
# compute node before exec'ing the real interpreter.
#
# wavlm.sh invokes every python step as "${python} -m <module> ...", so passing
#   --python local/bench/python_with_powerlog.sh
# wraps the training process without touching espnet itself. Only the process
# that actually trains is instrumented; short helper invocations exec straight
# through.
set -u

REAL_PYTHON=${BENCH_REAL_PYTHON:-python3}

# Set on the COMPUTE node: conf/slurm.conf submits with `sbatch --export=PATH`,
# so nothing exported by the caller survives. The 48M point OOM'd with 6.18 GiB
# "reserved but unallocated" while failing a 4.99 GiB request -- i.e. allocator
# fragmentation, not a real capacity wall. expandable_segments lets the caching
# allocator grow segments instead of stranding them.
# PYTORCH_CUDA_ALLOC_CONF is deprecated in favour of PYTORCH_ALLOC_CONF;
# set both so this keeps working across torch versions.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
OUTDIR=${BENCH_POWER_DIR:-$(pwd)/local/bench/power}
INTERVAL=${BENCH_POWER_INTERVAL:-5}

case " $* " in
    *" espnet2.bin.wavlm_train "*)
        # `nvidia-smi -L` also verifies the driver responds, which keeps the
        # login-node launcher process (which sees the inner training command line
        # after the "--") from starting a sampler that can only write an error.
        if [ "${BENCH_POWER:-1}" = "1" ] && nvidia-smi -L >/dev/null 2>&1; then
            mkdir -p "${OUTDIR}"
            stamp=$(date +%Y%m%d_%H%M%S)
            host=$(hostname -s)
            csv="${OUTDIR}/power_${host}_${stamp}.csv"
            echo "# nvidia-smi power sampling -> ${csv}" >&2
            nvidia-smi \
                --query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm \
                --format=csv,nounits \
                -l "${INTERVAL}" > "${csv}" 2>/dev/null &
            SAMPLER_PID=$!
            trap 'kill ${SAMPLER_PID} 2>/dev/null' EXIT INT TERM
        fi
        ;;
esac

"${REAL_PYTHON}" "$@"
exit $?
