#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Yifan Peng)

#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


inference_tag=

log "$0 $*"

. utils/parse_options.sh
. ./path.sh

inference_expdir="$1/${inference_tag}"
acc_file="${inference_expdir}/accuracy.csv"
echo "name,total,correct,accuracy" | tee ${acc_file}
for x in ${inference_expdir}/*; do
    if [ -d ${x} ]; then
        testset=$(basename ${x})
        python local/score.py --wer_dir "${x}/score_wer"
        echo "${testset},$(tail -n 1 ${x}/accuracy.csv)" | tee -a ${acc_file} || exit 1
    fi
done

echo "$0: Successfully wrote accuracy results to file ${acc_file}"
