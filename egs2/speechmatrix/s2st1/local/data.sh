#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

source ./scripts/utils/simple_dict.sh

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

# Full sequence of languages in the fixed order
full_langs=(cs de en es et fi fr hr hu it lt nl pl pt ro sk sl)
dict_init full_langs_indices
for i in "${!full_langs[@]}"; do
    dict_put full_langs_indices ${full_langs[$i]} $i;
done

src_langs=(lt) # one or many
tgt_langs=(en) # one or many

# Choose from flores and epst. 
# Notice: epst only covers de, en, es, fr, it, nl, pl, pt, ro
#         epst is s2t not s2s
# TODO: if src and tgt lang not in this set, directly use flores
test_dataset=epst


. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${SPEECH_MATRIX}
mkdir -p ${EUROPARL_ST}
# mkdir -p ${FLORES_ROOT}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"


# url for download FLORES data (for aligning speech in FLEURS with texts in FLORES)
europarl_raw_data_url=https://www.mllp.upv.es/europarl-st/v1.1.tar.gz
# flores_raw_data_url=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Download data to ${SPEECH_MATRIX} and ${EUROPARL_ST}"
    log "Prepare source aligned speech data from speech matrix for training"

    # # audio files for each languages
    # for lang in "${src_langs[@]}"; do
    #     mkdir -p ${SPEECH_MATRIX}/audios/${lang}
    #     local/download_and_unzip.sh \
    #         ${SPEECH_MATRIX}/audios/${lang} \
    #         https://dl.fbaipublicfiles.com/speech_matrix/audios/${lang}_aud.zip \
    #         ${lang}_aud.zip
    # done

    # for lang in "${tgt_langs[@]}"; do
    #     mkdir -p ${SPEECH_MATRIX}/audios/${lang}
    #     local/download_and_unzip.sh \
    #         ${SPEECH_MATRIX}/audios/${lang} \
    #         https://dl.fbaipublicfiles.com/speech_matrix/audios/${lang}_aud.zip \
    #         ${lang}_aud.zip
    # done

    # # Iterate over source and target languages
    # for src_lang in "${src_langs[@]}"; do
    #     for tgt_lang in "${tgt_langs[@]}"; do
    #         src_index=$(dict_get full_langs_indices "$src_lang")
    #         tgt_index=$(dict_get full_langs_indices "$tgt_lang")

    #         # Determine the pair order based on the indices
    #         if [ "$src_index" -lt "$tgt_index" ]; then
    #             pair="${src_lang}-${tgt_lang}"
    #         else
    #             pair="${tgt_lang}-${src_lang}"
    #         fi

    #         mkdir -p "${SPEECH_MATRIX}/aligned_speech/${pair}"

    #         local/download_and_unzip.sh \
    #             "${SPEECH_MATRIX}/aligned_speech/${pair}" \
    #             "https://dl.fbaipublicfiles.com/speech_matrix/aligned_speech/${pair}.tsv.gz" \
    #             "${pair}.tsv.gz"
    #     done
    # done

    # log "Download FLORES data to ${SPEECH_MATRIX}"
    # local/download_and_unzip.sh ${FLORES_ROOT} ${flores_raw_data_url} flores101_dataset.tar.gz

    log "Download EuroParl-ST data to ${EUROPARL_ST}"
    local/download_and_unzip.sh ${EUROPARL_ST} ${europarl_raw_data_url} v1.1.tar.gz
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: EUROPARL_ST data preparation"
    python local/fairseq_speechmatrix/prep_epst_test_data.py \
        --epst-dir ${EUROPARL_ST}/v1.1 \
        --proc-epst-dir ${EUROPARL_ST} \
        --save-root ${EUROPARL_ST}/test
    log "EUROPARL_ST data paraparation done."

    # log "Start fleurs data paraparation."
    # pip install datasets
    # python fairseq/examples/speech_matrix/valid_test_sets/preproc_fleurs_data.py \
    #     --proc-fleurs-dir ${FLORES_ROOT} > /dev/null 2>&1

    # log "Start align fleur data."
    # python fairseq/examples/speech_matrix/valid_test_sets/align_fleurs_data.py \
    #     --flores-root ${FLORES_ROOT}/flores101_dataset \
    #     --proc-fleurs-dir ${FLORES_ROOT} \
    #     --save-root ${FLORES_ROOT}/align > /dev/null 2>&1
    # log "Fleurs data alignment done."

    # python fairseq/examples/speech_matrix/valid_test_sets/prep_fleurs_test_data.py  \
    #     --proc-fleurs-dir ${FLORES_ROOT} \
    #     --save-root ${FLORES_ROOT}/test > /dev/null 2>&1
    # log "Fleurs data paraparation done."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Preparing data for speechmatrix"

    # install missing packages for functions in data_prep.py
    pip install bitarray

    for part in "train" "test" "dev"; do
        log "Preparing ${part} data."
        python local/data_prep.py \
            --src_folder "${SPEECH_MATRIX}" \
            --src_langs "${src_langs[@]}" \
            --tgt_langs "${tgt_langs[@]}" \
            --subset ${part} \
            --test_dataset "${test_dataset}" \
            --save_folder "data"


        for src_lang in "${src_langs[@]}"; do
            for tgt_lang in "${tgt_langs[@]}"; do
                # Skip if source language is the same as target language
                if [[ "$src_lang" == "$tgt_lang" ]]; then
                    continue
                fi

                ln -sf text.${tgt_lang} data/${part}_${src_lang}_${tgt_lang}/text
                ln -sf wav.scp.${tgt_lang} data/${part}_${src_lang}_${tgt_lang}/wav.scp

                utt_extra_files="wav.scp.${src_lang} wav.scp.${tgt_lang} text.${src_lang} text.${tgt_lang}"
                utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" data/${part}_${src_lang}_${tgt_lang}
            done
        done

    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"