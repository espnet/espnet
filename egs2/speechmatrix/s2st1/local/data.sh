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

langs=(cs de en es et fi fr hr hu it lt nl pl pt ro sk sl)
# langs=(lt sl)

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


# url for download FLORES data (for aligning speech in FLEURS with texts in FLORES)
flores_raw_data_url=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
europarl_raw_data_url=https://www.mllp.upv.es/europarl-st/v1.1.tar.gz

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${SPEECH_MATRIX}"
    log "Prepare source aligned speech data from speech matrix for training"

    # audio files for each languages
    for lang in "${langs[@]}"; do

        mkdir -p ${SPEECH_MATRIX}/audios/${lang}

        local/download_and_unzip.sh \
            ${SPEECH_MATRIX}/audios/${lang} \
            https://dl.fbaipublicfiles.com/speech_matrix/audios/${lang}_aud.zip \
            ${lang}_aud.zip
    done

    # audio alignments for each language pairs
    for i in {0..1}; do
        for j in {0..1}; do
            if [[ $i < $j ]]; then

                mkdir -p ${SPEECH_MATRIX}/aligned_speech/${langs[i]}-${langs[j]}

                local/download_and_unzip.sh \
                    ${SPEECH_MATRIX}/aligned_speech/${langs[i]}-${langs[j]} \
                    https://dl.fbaipublicfiles.com/speech_matrix/aligned_speech/${langs[i]}-${langs[j]}.tsv.gz \
                    ${langs[i]}-${langs[j]}.tsv.gz
            fi
        done
    done

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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for speechmatrix"
    ### Task dependent. You have to make data the following preparation part by yourself.

    for part in "train" "test" "dev"; do

        if [ "${part}" = train ]; then
            # install missing packages for functions in data_prep.py
            pip install bitarray
            python local/data_prep.py \
                --src_folder "${SPEECH_MATRIX}" \
                --langs "${langs[@]}" \
                --subset ${part} \
                --tgt "data"

        else
            log "to be updated"
        fi
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"