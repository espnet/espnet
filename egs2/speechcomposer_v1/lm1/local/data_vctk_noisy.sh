#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 1 ]; then
    log "Error: data_dir required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${NOISY_SPEECH}" ] ; then
    log "
    Please fill the value of 'NOISY_SPEECH' in db.sh
    The 'NOISY_SPEECH' (https://doi.org/10.7488/ds/2117) directory
    should at least contain the noisy speech and the clean reference:
        noisy_speech
        ├── clean_testset_wav
        ├── clean_trainset_28spk_wav
        ├── noisy_testset_wav
        └── noisy_trainset_28spk_wav
    "
    exit 1
fi

db_root=${NOISY_SPEECH}

train_set=train
dev_set=
eval_set=

vctk_noisy_dir="local/vctk_noisy"
data_dir=$1
data_dir="${data_dir}/se"
mkdir -p ${data_dir}


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    ${vctk_noisy_dir}/vctk_data_prep.sh  ${NOISY_SPEECH} ${data_dir} || exit 1;
fi
