#!/usr/bin/env bash
# Reuse the common ESPnet2 recipe infrastructure.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target-dir>" >&2
    exit 2
fi
dir=$1
mkdir -p "${dir}"
if [ ! -d "${dir}/../../TEMPLATE/asr1" ]; then
    echo "Error: ${dir}/../../TEMPLATE/asr1 must exist" >&2
    exit 1
fi
# The generic recipe infrastructure is shared with asr1; only aqa.sh is task-specific.
# path.sh resolves MAIN_ROOT from the generated recipe directory ($PWD).
for f in cmd.sh conf local; do
    cp -r "${dir}/../../TEMPLATE/asr1/${f}" "${dir}"
done
for f in path.sh db.sh scripts pyscripts steps utils; do
    ln -sfn "../../TEMPLATE/asr1/${f}" "${dir}/${f}"
done
ln -sfn ../../TEMPLATE/aqa1/aqa.sh "${dir}/aqa.sh"
