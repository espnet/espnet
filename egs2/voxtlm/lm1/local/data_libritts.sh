#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1
trim_all_silence=true

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 1 ]; then
    log "Error: data_dir is needed."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${LIBRITTS}" ]; then
   log "Fill the value of 'LIBRITTS' of db.sh"
   exit 1
fi
db_root=${LIBRITTS}
data_url=www.openslr.org/resources/60

lt_dir="local/libritts"
data_dir=$1
data_dir="${data_dir}/tts"

mkdir -p ${data_dir}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/donwload_and_untar.sh"
    # download the original corpus
    if [ ! -e "${db_root}"/LibriTTS/.complete ]; then
        for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
            ${lt_dir}/download_and_untar.sh "${db_root}" "${data_url}" "${part}"
        done
        touch "${db_root}/LibriTTS/.complete"
    else
        log "Already done. Skiped."
    fi

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"

    for name in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
        # Create kaldi data directory with the original audio
        ${lt_dir}/data_prep.sh "${db_root}/LibriTTS/${name}" "${data_dir}/${name}"
        
        utils/fix_data_dir.sh "${data_dir}/${name}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/combine_data.sh"
    utils/combine_data.sh ${data_dir}/train ${data_dir}/train-clean-100 ${data_dir}/train-clean-360 ${data_dir}/train-other-500
    utils/combine_data.sh ${data_dir}/dev ${data_dir}/dev-clean ${data_dir}/dev-other
    utils/combine_data.sh ${data_dir}/test ${data_dir}/test-clean ${data_dir}/test-other
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
