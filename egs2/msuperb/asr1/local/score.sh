#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
asr_exp=$1

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MSUPERB}
if [ -z "${MSUPERB}" ]; then
    log "Fill the value of 'MSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "Linguistic scoring started"
log "$0 $*"


# if [ -n "${inference_config}" ]; then
#         inference_tag="$(basename "${inference_config}" .yaml)"
# else
#     inference_tag=inference
# fi

# inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

python local/multilingual_analysis.py \
    --dir ${asr_exp}

# for dset in ${test_sets}; do
#     for _type in "cer" "wer"; do
#         _scoredir="${asr_exp}/${inference_tag}/${dset}/score_${_type}"

#         python local/multilingual_analysis.py \
#             --dir ${dir}
#     done
# done
