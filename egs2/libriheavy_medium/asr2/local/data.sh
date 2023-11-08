#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=5000
data_dir=data

log "$0 $*"
. utils/parse_options.sh

# url for the official repo.
libriheavy_repo=https://github.com/k2-fsa/libriheavy.git

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRILIGHT' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -d "${LIBRILIGHT}/small" ] && [ -d "${LIBRILIGHT}/medium" ] && [ -d "${LIBRILIGHT}/large" ]; then
        log "Libri-light found in ${LIBRILIGHT}."
        rm -fr libriheavy
        git clone $libriheavy_repo
    else
        echo "Some of ${LIBRILIGHT}/{small,medium,large} directories do not exist."
        echo "Please follow this link and download the audio data to ${LIBRILIGHT}:"
        echo "https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1a-downloading"
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Downloading and extracting the Libriheavy manifests"
    pushd libriheavy
    bash run.sh --stage 1 --stop-stage 2
    popd
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Preparing data directories in ${data_dir}"

    mkdir ${data_dir}

    for sub in small medium large test_clean test_other dev; do
        cp -a libriheavy/cases_and_punc/kaldi/${sub} ${data_dir}/${sub}
        sed -i 's! download/librilight! '"${LIBRILIGHT}"'!g' ${data_dir}/${sub}/wav.scp
        awk '{print $1" spk";}' ${data_dir}/${sub}/text > ${data_dir}/${sub}/utt2spk
        utils/fix_data_dir.sh ${data_dir}/${sub}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Combining training data directories"

    utils/combine_data.sh ${data_dir}/train_large ${data_dir}/large ${data_dir}/medium ${data_dir}/small
    utils/combine_data.sh ${data_dir}/train_medium ${data_dir}/medium ${data_dir}/small
    cp -a ${data_dir}/small ${data_dir}/train_small
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
