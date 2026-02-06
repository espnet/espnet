#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100
fs=24000

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${GTSINGER}" ]; then
    log "Fill the value of 'GTSINGER' of db.sh"
    exit 1
fi

mkdir -p ${GTSINGER}

train_set="tr_no_dev"
train_dev="dev"
recog_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The GTSinger data should be downloaded from https://huggingface.co/datasets/GTSinger/GTSinger
    # with authentication
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    # We use a pre-defined split (see details in local/dataset_split.py)"
    python local/dataset_split.py ${GTSINGER} \
        data/${train_set} data/${train_dev} data/${recog_set} --fs ${fs}

    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${src_data}
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
