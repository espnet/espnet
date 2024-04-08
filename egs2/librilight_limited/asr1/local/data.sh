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
SECONDS=0


stage=1
stop_stage=100000
ll_data_url=https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
ls_data_url=www.openslr.org/resources/12
train_set="train_10h"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRILIGHT_LIMITED}" ]; then
    log "Fill the value of 'LIBRILIGHT_LIMITED' of db.sh"
    exit 1
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

src=${LIBRILIGHT_LIMITED}/librispeech_finetuning

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data Download to ${LIBRILIGHT_LIMITED}"
    if [ ! -d ${src}/1h ] && [ ! -d ${src}/9h ]; then
        mkdir -p "${LIBRILIGHT_LIMITED}"
        wget "${ll_data_url}" -P "${LIBRILIGHT_LIMITED}"
        tar vxfz "${LIBRILIGHT_LIMITED}/librispeech_finetuning.tgz" -C "${LIBRILIGHT_LIMITED}"
    else
        log "${LIBRILIGHT_LIMITED}/librispeech_finetuning is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in 1h/{0..5}/{clean,other} 9h/{clean,other}; do
        dataname=$(echo ${part} | sed 's/\//_/g')

        data_part=$(./utils/make_absolute.sh ${src}/${part})
        data_new_path="data/train_${dataname}"
        mkdir -p "${data_new_path}"
        files=( "$(find -L ${data_part}/ -name '*.flac')" )
        # shellcheck disable=SC2068
        for wave_f in ${files[@]}; do
            filename=$(basename ${wave_f})
            filename=${filename%%.flac}
            echo "${filename} flac -c -d -s ${wave_f} |"
        done | sort | uniq > ${data_new_path}/wav.scp

        paste -d' ' <(awk '{print $1}' ${data_new_path}/wav.scp) \
            <(awk '{print $1}' "${data_new_path}/wav.scp" | cut -d'-' -f1) \
            > "${data_new_path}/utt2spk"
        ./utils/utt2spk_to_spk2utt.pl "${data_new_path}/utt2spk" > "${data_new_path}/spk2utt"
        text_files=( "$(find -L ${data_part}/ -name '*.trans.txt')" )
        # shellcheck disable=SC2068
        for text_f in ${text_files[@]}; do
            cat ${text_f}
        done | sort | uniq > "${data_new_path}/text"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine 10hr training sets"
    ./utils/combine_data.sh \
        data/${train_set} data/train_1h_{0..5}_{clean,other} data/train_9h_{clean,other}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
        echo "stage 4: Evaluation Data Download to ${LIBRISPEECH}"
        for part in dev-clean test-clean dev-other test-other; do
            local/download_and_untar_eval.sh "${LIBRISPEECH}" "${ls_data_url}" "${part}"
        done
    else
        log "stage 4: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Evaluation Data Preparation and Combination"
    for part in dev-clean test-clean dev-other test-other; do
        # use underscore-separated names in data directories.
        local/data_prep_eval.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${part//-/_}
    done
    utils/combine_data.sh --extra_files utt2num_frames data/dev data/dev_clean data/dev_other
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
