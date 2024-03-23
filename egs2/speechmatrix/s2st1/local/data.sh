#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
src_lang=lt # cs de en es et fi fr hr hu it lt nl pl pt ro sk sl
dst_lang=sl # cs de en es et fi fr hr hu it lt nl pl pt ro sk sl

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${SPEECH_MATRIX}
if [ -z "${SPEECH_MATRIX}" ]; then
    log "Fill the value of 'SPEECH_MATRIX' of db.sh"
    exit 1
fi

FLORES_ROOT="${SPEECH_MATRIX}/flores"
mkdir -p ${FLORES_ROOT}
EPST_DIR="${SPEECH_MATRIX}/epst"
mkdir -p ${EPST_DIR}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

# base url for download speech_matrix data
# url 1 for source language and url 2 for target language
# aligned data tsv indicates the aligned utts
speech_matrix_raw_data_url_1=https://dl.fbaipublicfiles.com/speech_matrix/audios/${src_lang}_aud.zip
speech_matrix_raw_data_url_2=https://dl.fbaipublicfiles.com/speech_matrix/audios/${dst_lang}_aud.zip
speech_matrix_aligned_data_tsv=https://dl.fbaipublicfiles.com/speech_matrix/aligned_speech/${src_lang}-${dst_lang}.tsv.gz

# url for download FLORES data (for aligning speech in FLEURS with texts in FLORES)
flores_raw_data_url=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
europarl_raw_data_url=https://www.mllp.upv.es/europarl-st/v1.1.tar.gz

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${SPEECH_MATRIX}"
    log "Prepare source aligned speech data from speech matrix for training"
    mkdir -p ${SPEECH_MATRIX}/audios/${src_lang}
    local/download_and_unzip.sh ${SPEECH_MATRIX}/audios/${src_lang} ${speech_matrix_raw_data_url_1} ${src_lang}_aud.zip
    mkdir -p ${SPEECH_MATRIX}/audios/${dst_lang}
    local/download_and_unzip.sh ${SPEECH_MATRIX}/audios/${dst_lang} ${speech_matrix_raw_data_url_2} ${dst_lang}_aud.zip
    mkdir -p ${SPEECH_MATRIX}/aligned_speech/${src_lang}-${dst_lang}
    local/download_and_unzip.sh ${SPEECH_MATRIX}/aligned_speech/${src_lang}-${dst_lang} ${speech_matrix_aligned_data_tsv} ${src_lang}-${dst_lang}.tsv.gz

    log "Download FLORES data to ${SPEECH_MATRIX}"
    local/download_and_unzip.sh ${FLORES_ROOT} ${flores_raw_data_url} flores101_dataset.tar.gz
    log "Download EuroParl-ST data to ${SPEECH_MATRIX}"
    local/download_and_unzip.sh ${EPST_DIR} ${europarl_raw_data_url} v1.1.tar.gz

    log "Install Fairseq package for preparing valid and test data"
    git clone --branch ust --depth 1 https://github.com/facebookresearch/fairseq.git

    # Temparally change the PYTHONPATH for running fairseq python scripts
    (export PYTHONPATH="${PYTHONPATH}:$(pwd)/fairseq/"

    log "Start epst data paraparation."
    pip install num2words    
    python fairseq/examples/speech_matrix/valid_test_sets/prep_epst_test_data.py \
        --epst-dir ${EPST_DIR}/v1.1 \
        --proc-epst-dir ${EPST_DIR} \
        --save-root ${EPST_DIR}/test

    log "Epst data paraparation done."
    
    log "Start fleurs data paraparation."
    # check the fairseq/examples/speech_matrix/valid_test_sets/preproc_fleurs_data.py, change 
    # "data = load_dataset("fleurs", lang)" in line 12 to "data = load_dataset("google/fleurs", lang)"
    pip install datasets
    python fairseq/examples/speech_matrix/valid_test_sets/preproc_fleurs_data.py \
        --proc-fleurs-dir ${FLORES_ROOT}

    log "Start align fleur data."
    python fairseq/examples/speech_matrix/valid_test_sets/align_fleurs_data.py \
        --flores-root ${FLORES_ROOT}/flores101_dataset \
        --proc-fleurs-dir ${FLORES_ROOT} \
        --save-root ${FLORES_ROOT}/align 
    log "Fleurs data alignment done."

    python fairseq/examples/speech_matrix/valid_test_sets/prep_fleurs_test_data.py  \
        --proc-fleurs-dir ${FLORES_ROOT} \
        --save-root ${FLORES_ROOT}/test
    log "Fleurs data paraparation done."
    )
fi

log "Successfully finished. [elapsed=${SECONDS}s]"