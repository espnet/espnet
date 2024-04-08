#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


help_message=$(cat << EOF
Usage: $0
  optional argument:
    None
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;



. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

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

log "Data preparation"
# The following datasets will be created:
# tr_26spk, {cv,tt}_2spk
local/vctk_data_prep.sh  ${NOISY_SPEECH} || exit 1;
