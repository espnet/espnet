#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=-1
stop_stage=2
use_speakeropen=false

help_message=$(cat << EOF
Usage: $0 
  optional argument:
    None
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if "${use_speakeropen}"; then

    if [ ! -e "${NOISY_SPEECH}" ] ; then
        log "
        Please fill the value of 'NOISY_SPEECH' in db.sh
        The 'NOISY_SPEECH' (https://doi.org/10.7488/ds/2117) directory
        should at least contain the clean speech and the clean text:
            noisy_speech
            ├── clean_testset_wav
            ├── clean_trainset_28spk_wav
            ├── testset_txt
            └── trainset_28spk_txt
        "
	exit 1
    fi

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	log "stage 0: local/data_prep_speaker_open.sh"
	# Initial normalization of the data
	# Doesn't change sampling frequency and it's done after stages
    local/data_prep_speaker_open.sh  ${NOISY_SPEECH} || exit 1;
    fi

else
    
    if [ -z "${VCTK}" ]; then
	log "Please fill the value of 'VCTK' of db.sh"
	exit 1
    fi
    db_root=${VCTK}
    
    train_set=tr_no_dev
    dev_set=dev
    eval_set=eval1

    if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
	log "stage -1: Data Download"
	local/data_download.sh "${db_root}"
    fi
    
    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	log "stage 0: local/data_prep_speaker_closed.sh"
	# Initial normalization of the data
	# Doesn't change sampling frequency and it's done after stages
	local/data_prep_speaker_closed.sh \
            --train_set "${train_set}" \
            --dev_set "${dev_set}" \
            --eval_set "${eval_set}" \
            "${db_root}"/VCTK-Corpus
    fi

fi
