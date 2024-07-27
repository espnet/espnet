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
stop_stage=3

librilight_data_url="https://dl.fbaipublicfiles.com/librilight/data"
librilight_parts="small medium large"
train_set="train"
train_dev="dev"
train_eval="test"

segment_dir=        # output dir of LibriLight VAD segmentation
nj=32

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 1 ]; then
    log "Error: data_dir required."
    exit 2
fi

if [ -z "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRISPEECH' and 'LIBRILIGHT' of db.sh"
    exit 1
fi

data_dir=$1
data_dir="${data_dir}/speechlm"
data_dir_librispeech_asr="data/librispeech/asr"
mkdir -p ${data_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${LIBRILIGHT}/.complete" ]; then
        echo "Stage 1b: Data Download librilight data to ${LIBRILIGHT}"
        for part in ${librilight_parts}; do
            local/librilight/download_and_untar_librilight.sh ${LIBRILIGHT} ${librilight_data_url} ${part}
        done
    else
        log "stage 1b: ${LIBRILIGHT}/.complete is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in ${librilight_parts}; do
		log "Segment ${LIBRILIGHT}/${part} to ${LIBRILIGHT}/${part}_segmented"

        if [ -z ${segment_dir} ]; then
            output_dir=${LIBRILIGHT}/${part}_segmented
        else
            output_dir=${segment_dir}/${part}_segmented
        fi

        _logdir=${output_dir}/logdir
        mkdir -p ${_logdir}

        ${train_cmd} --num_threads ${nj} "${_logdir}/segment_audio.log" \
            python local/librilight/cut_by_vad.py \
                --input_dir "${LIBRILIGHT}/${part}" \
                --output_dir "${output_dir}" \
                --target_len_sec 15 \
                --n_workers ${nj} \
                --out_extension ".flac"

        local/librilight/data_prep_librilight.sh ${output_dir} ${data_dir}/librilight_${part}
    done

    log "combine all training and development sets"
    utils/combine_data.sh ${data_dir}/${train_set} ${data_dir}/librilight_small ${data_dir}/librilight_medium {data_dir}/librilight_large

    # copy dev from Librispeech
    mkdir -p ${data_dir}/${train_dev}
    utils/copy_data_dir.sh ${data_dir_librispeech_asr}/dev ${data_dir}/${train_dev}
    rm ${data_dir}/${train_dev}/text

    # copy eval from Librispeech
    mkdir -p ${data_dir}/"test"
    utils/copy_data_dir.sh ${data_dir_librispeech_asr}/test ${data_dir}/"test"
    rm ${data_dir}/"test"/text

   # data prep for Librispeech
    for part in dev-clean dev-other test-clean test-other; do
        # use underscore-separated names in data directories.
        local/librilight/data_prep_librispeech.sh ${LIBRISPEECH}/LibriSpeech/${part} ${data_dir}/${part//-/_}
    done
    log "combine all test and development sets"
    utils/combine_data.sh  ${data_dir}/${train_dev} ${data_dir}/dev_clean ${data_dir}/dev_other
    utils/combine_data.sh ${data_dir}/${train_eval} ${data_dir}/test_clean ${data_dir}/test_other

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
