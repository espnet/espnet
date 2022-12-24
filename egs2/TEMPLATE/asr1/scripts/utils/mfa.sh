#!/usr/bin/env bash

# Copyright 2022 Hitachi LTD. (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Preparation of data and generation of MFA alignments
# You need to install the following tools to run this script:
# $ conda config --append channels conda-forge
# $ conda install montreal-forced-aligner

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

# General configuration
stage=0
stop_stage=100
nj=12
workdir=data/local/mfa  
clean_temp=false
split_sets=

# Data prep related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
fs=22050

# MFA/Tokenization related
lang="english_us_tacotron"
acoustic_model="english_mfa"
dictionary="english_us_mfa"
g2p_model="english_us_mfa"
cleaner=tacotron
train=false
max_phonemes_word=7  # split the phonemes durations for word durations. (Ref. PortaSpeech)

help_message=$(cat << EOF
Usage: $0 --stage "<stage>" --stop-stage "<stop_stage>" --fs "<fs>"

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage           # Processes is stopped at the specified stage (default="${stop_stage}").
    --nj                   # The number of parallel jobs (default="${nj}").
    --workdir
    --clean_temp
    --datasets

    # Data prep related
    --local_data_opts      # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --fs                   # Sampling rate (default="${fs}").

    # Tokenization related
    --lang
    --g2p
    --train
    --max_phonemes_word

EOF
)

log "$0 $*"

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

if [[ "$(basename "$(pwd)")" != tts* ]]; then
    log "Error: You must cd to a tts directory"
    exit 1
fi

if [ -z "${split_sets}" ]; then
    log "Error: You need to add the split sets with --split_sets <train> <dev> <tests>"
    log "Error: Check you 'local/data.sh' or 'run.sh' file to get the name of the split sets."
    exit 1
fi

if ! [ -x "$(command -v mfa)" ]; then
    log "ERROR: Missing mfa, run 'cd ../../../tools; make mfa.done; cd -;'"
    exit 1
fi

mkdir -p ${workdir}
tempdir=${workdir}/tmp
corpus_dir=${workdir}/corpus

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # [Task dependent] Need to create data.sh for new corpus
    log "Stage 0: Data preparation for ${split_sets}"
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Prepare Data set for MFA"
    # Text cleaning and save it in wav/lab files (by file)
    python pyscripts/utils/mfa_format.py labs \
                                    --data_sets "${split_sets}" \
                                    --save_dir "${corpus_dir}"

fi

if ${train}; then
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    
        log "stage 2: Training MFA"

        # Generate dictionary using ESPnet TTS frontend
        python pyscripts/utils/mfa_format.py dictionary \
            --corpus_dir "${corpus_dir}" \
            --save_dict "${workdir}/lexicon.txt"

        # Train G2P
        mfa train_g2p --clean\
            --phonetisaurus \
            -t ${tempdir} \
            ${workdir}/train_dict.txt ${workdir}/${lang}.zip

        # Train G2P
        mfa g2p --clean -t ${tempdir} \
            ${workdir}/${lang}.zip \
            ${corpus_dir} \
            ${workdir}/lexicon.txt

        mfa train -t ${tempdir} \
            ${corpus_dir} \
            ${workdir}/lexicon.txt \
            ${workdir}/acoustic_model.zip
    fi

    # # create OOV dictionary
    # set +e
    # mfa validate "${wavs_dir}" "${dictionary}" "${acoustic_model}" --brackets '' | while read -r line; do
    #     if [[ $line =~ "jobs" ]]; then
    #         echo "OOV file created, stopping MFA."
    #         pkill mfa
    #     else
    #         echo "$line"
    #     fi
    # done
    # set -e
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "stage 3: Generating aligments using MFA model"

            # # Generate a text using espnet2-based g2p
            # python pyscripts/mfa/reformat_dict.py \
            #         --g2p ${g2p} \
            #         ${workdir}/list2.txt.tmp
            lexicon=${workdir}/lexicon.txt
            acoustic=${workdir}/acoustic_model.zip

            # Remove punctuation and clitic from aligment, otherwise it will generate a issue with g2p model
            echo "punctuation: null" > "${workdir}"/config.yaml
            echo "clitic_markers: null" >> "${workdir}"/config.yaml


        mfa align -j ${nj} \
            --clean \
            -t ${tempdir} \
            --config_path "${workdir}"/config.yaml \
            --output_format json \
            ${corpus_dir} \
            ${lexicon} \
            ${acoustic} \
            ${workdir}/alignments

        if ${clean_temp}; then
            # Be careful, this will delete all the files employed for training the mfa's models.
            rm -rf ${tempdir}
        fi
    fi
else
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "stage 2: Download pretrained MFA models"
        # download pretrained MFA models
        mfa models download acoustic "${acoustic_model}"
        mfa models download dictionary "${dictionary}"
        mfa models download g2p "${g2p_model}"
    fi

    # # create OOV dictionary
    # set +e
    # mfa validate "${wavs_dir}" "${dictionary}" "${acoustic_model}" --brackets '' | while read -r line; do
    #     if [[ $line =~ "jobs" ]]; then
    #         echo "OOV file created, stopping MFA."
    #         pkill mfa
    #     else
    #         echo "$line"
    #     fi
    # done
    # set -e

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "stage 3: Generating aligments using MFA model"

        mfa validate --clean -j ${nj} \
                -t ${tempdir} \
                ${corpus_dir} \
                ${lexicon} \
                ${acoustic}
        
        mfa align -j ${nj} \
                --clean \
                -t ${tempdir} \
                --config_path "${workdir}"/config.yaml \
                --output_format json \
                ${corpus_dir} \
                ${lexicon} \
                ${acoustic} \
                ${workdir}/alignments
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare phoneme-text labels"

    echo "<sil>" > data/local/nlsyms.txt

    python local/get_phones_alignments.py \
        --samplerate ${fs} \
        --g2p ${g2p} \
        --max_phonemes_word ${max_phonemes_word} \
        ${workdir}/alignments \
        ${workdir}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Prepare data sets with phoneme alignments"
    for dset in ${split_sets}; do
        utils/copy_data_dir.sh data/"${dset}"{,_phn}
        cp ${workdir}/text.phn data/${dset}_phn/text
        utils/fix_data_dir.sh data/${dset}_phn

        utils/filter_scp.pl data/${dset}_phn/utt2spk ${workdir}/durations > data/${dset}_phn/durations
        utils/filter_scp.pl data/${dset}_phn/utt2spk ${workdir}/word_durations > data/${dset}_phn/word_durations
    done
fi

# mfa g2p "${g2p_model}" "${mfa_dir}/wavs_validate_pretrained/oovs_found_${dictionary}.txt" "${oov_dict}"
# cat "${dict_dir}/${dictionary}.dict" "${oov_dict}" > "${dict_dir}/${dictionary}_${dataset}.dict"

# # check again
# mfa validate "${wavs_dir}" "${dictionary}_${dataset}" "${acoustic_model}" --brackets ''

# # perform force alignment
# mfa align "${wavs_dir}" "${dictionary}_${dataset}" "${acoustic_model}" ./textgrids

echo "Successfully finished generating data and MFA alignments."

# NOTE(iamanigeeit): If you want to train FastSpeech2 with the alignments,
#   please check `egs2/ljspeech/tts1/local/run_mfa.sh`. For example:
# $ ./local/run_mfa.sh --stage 0 --stop_stage 0
# $ ./local/run_mfa.sh --stage 1 --stop_stage 1
# $ ./local/run_mfa.sh --stage 2 --stop_stage 2
# $ ./local/run_mfa.sh --stage 3 --stop_stage 3
# $ ./local/run_mfa.sh --stage 4 --stop_stage 4
# $ ./local/run_mfa.sh --stage 5 --stop_stage 5 \
#     --train_config conf/tuning/train_fastspeech2.yaml \
#     --teacher_dumpdir data \
#     --tts_stats_dir data/stats \
#     --write_collected_feats true
# $ ./local/run_mfa.sh --stage 6 --stop_stage 6 \
#     --train_config conf/tuning/train_fastspeech2.yaml \
#     --teacher_dumpdir data \
#     --tts_stats_dir data/stats
