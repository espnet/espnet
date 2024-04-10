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

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

FLORES=downloads/flores
mkdir -p ${FLORES}

# defined in db.sh
mkdir -p ${SPEECH_MATRIX}
mkdir -p ${EUROPARL_ST}
mkdir -p ${FLEURS}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"


# url for download FLORES data (for aligning speech in FLEURS with texts in FLORES)
europarl_raw_data_url=https://www.mllp.upv.es/europarl-st/v1.1.tar.gz
flores_raw_data_url=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
kmeans_model_url=https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
hifigan_ckpt_url=https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000
hifigan_config_url=https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Download data to ${SPEECH_MATRIX} and ${EUROPARL_ST}"
    log "Prepare source aligned speech data from speech matrix for training"

    mkdir -p ${SPEECH_MATRIX}/audios
    # audio files for each languages
    for lang in "${src_langs[@]}"; do
        local/download_and_unzip.sh \
            --skip-unzip \
            ${SPEECH_MATRIX}/audios \
            https://dl.fbaipublicfiles.com/speech_matrix/audios/${lang}_aud.zip \
            ${lang}_aud.zip
    done

    for lang in "${tgt_langs[@]}"; do
        local/download_and_unzip.sh \
            --skip-unzip \
            ${SPEECH_MATRIX}/audios \
            https://dl.fbaipublicfiles.com/speech_matrix/audios/${lang}_aud.zip \
            ${lang}_aud.zip
    done

    # Iterate over source and target languages
    for src_lang in "${src_langs[@]}"; do
        for tgt_lang in "${tgt_langs[@]}"; do
            src_index=$(dict_get full_langs_indices "$src_lang")
            tgt_index=$(dict_get full_langs_indices "$tgt_lang")

            # Determine the pair order based on the indices
            if [ "$src_index" -lt "$tgt_index" ]; then
                pair="${src_lang}-${tgt_lang}"
            else
                pair="${tgt_lang}-${src_lang}"
            fi

            mkdir -p "${SPEECH_MATRIX}/aligned_speech/${pair}"

            local/download_and_unzip.sh \
                --skip-unzip \
                ${SPEECH_MATRIX}/aligned_speech/${pair} \
                https://dl.fbaipublicfiles.com/speech_matrix/aligned_speech/${pair}.tsv.gz \
                ${pair}.tsv.gz
        done
    done

    log "Download FLORES data to ./data"
    local/download_and_unzip.sh --remove-archive ${FLORES} ${flores_raw_data_url} flores101_dataset.tar.gz

    log "Download EuroParl-ST data to ${EUROPARL_ST}"
    local/download_and_unzip.sh --remove-archive ${EUROPARL_ST} ${europarl_raw_data_url} v1.1.tar.gz

    log "Download pre-trained k-means model to dump/pretrained_kmeans"
    mkdir -p dump/pretrained_kmeans
    wget ${kmeans_model_url} -O dump/pretrained_kmeans/km_1000.mdl

    log "Download pre-trained HifiGAN ckpt and config" # only english for now
    mkdir -p dump/pretrained_hifigan
    wget ${hifigan_ckpt_url} -O dump/pretrained_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj.pt
    wget ${hifigan_config_url} -O dump/pretrained_hifigan/config.yml
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: EUROPARL_ST + FLEURS data preparation"

    log "EUROPARL_ST data preparation"
    python local/fairseq_speechmatrix/prep_epst_test_data.py \
        --epst-dir ${EUROPARL_ST}/v1.1 \
        --proc-epst-dir ${EUROPARL_ST} \
        --save-root ${EUROPARL_ST}
    log "EUROPARL_ST data preparation done."

    log "Start FLEURS data paraparation."
    python local/fairseq_speechmatrix/preproc_fleurs_data.py \
        --proc-fleurs-dir ${FLEURS}
    python local/fairseq_speechmatrix/align_fleurs_data.py \
        --flores-root ${FLORES}/flores101_dataset \
        --proc-fleurs-dir ${FLEURS} \
        --save-root ${FLEURS}
    log "FLEURS data preparation done."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Preparing data for speechmatrix"

    for part in "train" "dev" "test"; do
        log "Preparing ${part} data."
        python local/data_prep.py \
            --src_folder ${SPEECH_MATRIX} \
            --src_langs ${src_langs[@]} \
            --tgt_langs ${tgt_langs[@]} \
            --subset ${part} \
            --save_folder data \
            --dump_folder dump \
            --europarl_folder ${EUROPARL_ST} \
            --fleurs_folder ${FLEURS}

        for src_lang in "${src_langs[@]}"; do
            for tgt_lang in "${tgt_langs[@]}"; do
                # Skip if source language is the same as target language
                if [[ "$src_lang" == "$tgt_lang" ]]; then
                    continue
                fi

                # Solve train and dev
                data_path=data/${part}_${src_lang}_${tgt_lang}
                if [ ! -d ${data_path} ]; then
                    continue
                fi

                ln -sf text.${tgt_lang} ${data_path}/text
                ln -sf wav.scp.${tgt_lang} ${data_path}/wav.scp

                utt_extra_files="wav.scp.${src_lang} wav.scp.${tgt_lang} text.${src_lang} text.${tgt_lang}"
                utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${data_path}

                # Solve test
                for dataset in "epst" "fleurs"; do
                    data_path=data/${part}_${dataset}_${src_lang}_${tgt_lang}
                    if [ ! -d ${data_path} ]; then
                        continue
                    fi

                    ln -sf text.${tgt_lang} ${data_path}/text
                    ln -sf wav.scp.${tgt_lang} ${data_path}/wav.scp

                    utt_extra_files="wav.scp.${src_lang} wav.scp.${tgt_lang} text.${src_lang} text.${tgt_lang}"
                    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${data_path}
                done
            done
        done
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
