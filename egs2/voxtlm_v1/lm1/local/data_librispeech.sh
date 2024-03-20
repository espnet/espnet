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
data_url=www.openslr.org/resources/12
train_set="train"
train_dev="dev"
train_eval="test"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 1 ]; then
    log "Error: One argument is required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

ls_dir="local/librispeech"
data_dir=$1
data_dir_asr="${data_dir}/asr"
data_dir_textlm="${data_dir}/textlm"
mkdir -p ${data_dir_asr}
mkdir -p ${data_dir_textlm}


# ASR - data processing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
        echo "stage 1: Data Download to ${LIBRISPEECH}"
        for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
            ${ls_dir}/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
        done
    else
        log "stage 1: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        ${ls_dir}/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} ${data_dir_asr}/${part//-/_}
    done
    log "combine all training and development sets"
    utils/combine_data.sh ${data_dir_asr}/${train_set} ${data_dir_asr}/train_clean_100 ${data_dir_asr}/train_clean_360 ${data_dir_asr}/train_other_500
    utils/combine_data.sh ${data_dir_asr}/${train_dev} ${data_dir_asr}/dev_clean ${data_dir_asr}/dev_other
    utils/combine_data.sh ${data_dir_asr}/${train_eval} ${data_dir_asr}/test_clean ${data_dir_asr}/test_other

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p ${data_dir_textlm}/train
    # use external data
    if [ ! -e ${data_dir_textlm}/librispeech-lm-norm.txt.gz ]; then
        log "stage 4: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${data_dir_textlm}/train/
    fi
    if [ ! -e ${data_dir_textlm}/train/text ]; then
        # provide utterance id to each texts
        # e.g., librispeech_lng_00003686 A BANK CHECK
        zcat ${data_dir}/textlm/librispeech-lm-norm.txt.gz | \
            awk '{ printf("textlm_librispeech_lng_%08d %s\n",NR,$0) } ' > ${data_dir_textlm}/train/text
    fi

    # copy dev
    mkdir -p ${data_dir_textlm}/dev
    cp ${data_dir_asr}/dev_clean/text ${data_dir_textlm}/dev/text
    cat ${data_dir_asr}/dev_other/text >> ${data_dir_textlm}/dev/text
    sed -i "s/asr_/textlm_/" ${data_dir_textlm}/dev/text

    # copy tests
    mkdir -p ${data_dir_textlm}/test
    cp ${data_dir_asr}/test_clean/text ${data_dir_textlm}/test/text
    cat ${data_dir_asr}/test_other/text >> ${data_dir_textlm}/test/text
    sed -i "s/asr_/textlm_/" ${data_dir_textlm}/test/text
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
