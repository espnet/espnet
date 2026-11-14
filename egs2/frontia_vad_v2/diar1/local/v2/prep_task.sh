#!/usr/bin/env bash
# One task of stage 3: the corpora of a list whose line number is TASK modulo NTASKS, each
# through local/v2/prep.py, its NSHARDS shards in parallel in this task.
#   local/v2/prep_task.sh <corpus list> <task (1-based)> <ntasks> <root> <dumpdir> <nshards> <pack_seconds>
set -euo pipefail
list=$1; task=$2; ntasks=$3; root=$4; dumpdir=$5; nshards=$6; pack_seconds=$7
v2=$(dirname "$0")
awk -v t="${task}" -v n="${ntasks}" '(NR - 1) % n == t - 1' "${list}" | while read -r corpus; do
    if [ -f "${dumpdir}/${corpus}/.done" ]; then
        echo "skip ${corpus}: done"
        continue
    fi
    echo "$(date '+%F %T') ${corpus}"
    seq 0 $((nshards - 1)) | xargs -P "${nshards}" -I{} sh -c \
        "python3 ${v2}/prep.py --corpus ${corpus} --root ${root} --out ${dumpdir} \
            --shard {} --nshards ${nshards} --pack_seconds ${pack_seconds} \
            > ${dumpdir}/log/${corpus}.{}.log 2>&1 || { echo ${corpus} shard {} failed; exit 255; }"
    touch "${dumpdir}/${corpus}/.done"
    echo "$(date '+%F %T') ${corpus} done"
done
