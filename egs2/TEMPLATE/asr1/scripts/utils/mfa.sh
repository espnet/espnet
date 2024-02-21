#!/usr/bin/env bash

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Preparation of data and generation of MFA alignments
# You need to install the following tools to run this script:
# $ conda install -c conda-forge montreal-forced-aligner=2.2.17

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
stage=0                             # Processes starts from the specified stage.
stop_stage=100                      # Processes is stopped at the specified stage.
nj=12                               # The number of parallel jobs.
workdir=data/local/mfa              # Directory to save temporal files of MFA processing.
clean_temp=false                    # Whether to remove the temporal files at the end of the processing.
split_sets="tr_no_dev dev eval1"    # Name of the splits of the data that will be used during the whole processing.

# Data prep related
local_data_opts=""  # Options to be passed to local/data.sh.
hop_size=256        # The number of shift points.
samplerate=22050    # Sampling rate.

# MFA/Tokenization related
language=""                     # The language type of corpus.
acoustic_model="english_mfa"    # MFA Acoustic model.
dictionary="english_us_mfa"     # MFA Dictionary.
g2p_model="english_us_mfa"      # Grapheme-to-phoneme model. If train, g2p_model is from ESPnet, otherwise from MFA.
cleaner=mfa_english             # Text Cleaner.
train=false                     # Whether to train the MFA models (acoustic and g2p) or not.
mfa_validate=false              # Whether to run mfa validate before alignment. Disabled by default because it repeats g2p and alignment.
textgrid_format="json"          # MFA alignment output format.
single_speaker=false            # Whether corpus is single-speaker or not.
# max_phonemes_word=7  # split the phonemes durations for word durations. (Ref. PortaSpeech)

help_message=$(cat << EOF
Usage: $0 --stage "<stage>" --stop_stage "<stop_stage>" --train <train>

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage           # Processes is stopped at the specified stage (default="${stop_stage}").
    --nj                   # The number of parallel jobs (default="${nj}").
    --workdir              # Directory to save temporal files of MFA processing (default="${workdir}").
    --clean_temp           # Whether to remove the temporal files at the end of the processing (default="${clean_temp}").
    --split_sets           # Name of the splits of the data that will be used during the whole processing (default="${split_sets}").

    # Data prep related
    --local_data_opts      # Options to be passed to local/data.sh (default="${local_data_opts}").
    --hop_size             # The number of shift points (default="${hop_size}").
    --samplerate           # Sampling rate (default="${samplerate}").

    # MFA/Tokenization related
    --language             # The language type of corpus (default="${language}").
    --acoustic_model       # MFA Acoustic model (default="${acoustic_model}").
    --dictionary           # MFA Dictionary (default="${dictionary}").
    --g2p_model            # MFA Grapheme-to-phoneme model (default="${g2p_model}").
    --cleaner              # Text Cleaner (default="${cleaner}").
    --train                # Whether to train the MFA models (acoustic and g2p) or not (default="${train}").
    --mfa_validate         # Whether to run mfa validate before alignment. Disabled by default because it repeats g2p and alignment.
    --textgrid_format      # MFA alignment output format (default="${textgrid_format}").
    --single_speaker       # Whether corpus is single-speaker or not. (default="${single_speaker}").

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


if [ -z "${split_sets}" ]; then
    log "Error: You need to add the split sets with --split_sets <train> <dev> <tests>"
    log "Error: Check you 'local/data.sh' or 'run.sh' file to get the name of the split sets."
    exit 1
fi

if ! [ -x "$(command -v mfa)" ]; then
    log "ERROR: Missing mfa, run 'cd ../../../tools; make mfa.done; cd -;'"
    exit 1
fi

if [ -n "${language}" ]; then
    if [ -z "${acoustic_model}" ]; then
        acoustic_model="${language}"
    fi
    if [ -z "${dictionary}" ]; then
        dictionary="${language}"
    fi
    if [ -z "${g2p_model}" ]; then
        g2p_model="${language}"
    fi
fi

if [ -z "${acoustic_model}" ]; then
    log "ERROR: You need to add <language> or <acoustic_model>."
    exit 1
fi
if [ -z "${dictionary}" ]; then
    log "ERROR: You need to add <language> or <dictionary>."
    exit 1
fi
if [ -z "${g2p_model}" ]; then
    log "ERROR: You need to add <language> or <g2p_model>."
    exit 1
fi

mkdir -p "${workdir}"
tempdir="${workdir}/tmp"
corpus_dir="${workdir}/corpus"
oov_dict="${workdir}/oov_corpus.dict"

if ${train}; then
    dictionary=${language}
    dict_tag_or_path="${workdir}/${language}.txt"
    src_dict="${dict_tag_or_path}"
    acoustic_model="${workdir}/acoustic/${language}.zip"
    espnet_g2p="${g2p_model}"
    mfa_g2p="${workdir}/g2p/${language}.zip"
else
    dict_tag_or_path="${dictionary}"
    src_dict="${HOME}/Documents/MFA/pretrained_models/dictionary/${dictionary}.dict"
    espnet_g2p=none
    mfa_g2p="${g2p_model}"
fi

if ${single_speaker}; then
    single_spk_opt="--single_speaker"
else
    single_spk_opt=""
fi

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
                                    --text_cleaner "${cleaner}" \
                                    --g2p_model "${espnet_g2p}" \
                                    --corpus_dir "${corpus_dir}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if ${train}; then
        log "stage 2: Training MFA"
        mkdir -p ${workdir}/{g2p,acoustic}

        # Generate dictionary using ESPnet TTS frontend
        log "Generating training dictionary with ESPnet G2P..."
        # shellcheck disable=SC2154
        ${train_cmd} ${tempdir}/logs/dict_g2p.log \
            python pyscripts/utils/mfa_format.py dictionary \
                --corpus_dir "${corpus_dir}" \
                --g2p_model "${espnet_g2p}"

        # Train G2P
        log "Training G2P model with custom dictionary."
        ${train_cmd} ${tempdir}/logs/train_g2p.log \
            mfa train_g2p -j ${nj} \
                --clean \
                --phonetisaurus \
                -t ${tempdir} \
                ${workdir}/train_dict.txt \
                "${mfa_g2p}"

        # Generate lexicon
        log "Generating Dictionary..."
        ${train_cmd} ${tempdir}/logs/generate_dict.log \
            mfa g2p \
                --clean \
                -j ${nj} \
                -t ${tempdir} \
                "${corpus_dir}" \
                "${mfa_g2p}" \
                "${dict_tag_or_path}"

        # Train MFA. Kaldi fails with relative paths
        log "Training MFA model..."
        ${train_cmd} ${tempdir}/logs/train_align.log \
            mfa train \
                -j ${nj} \
                -t "$(pwd)/${tempdir}" \
                ${single_spk_opt} \
                "${corpus_dir}" \
                "${dict_tag_or_path}" \
                "${acoustic_model}"

    elif [[ ${language} == espnet* ]]; then
        # TODO(fhrozen): Upload models to huggingface
        log "stage 2: Download pretrained MFA models from Huggingface"
        log "ERROR: WIP"
        exit 1
    else
        log "stage 2: Download pretrained MFA models"
        # download pretrained MFA models
        mfa models download acoustic "${acoustic_model}"
        mfa models download dictionary "${dictionary}"
        mfa models download g2p "${mfa_g2p}"
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Generating OOV Dictionary"

    log "Generating Dictionary & OOV Dictionary..."
    # create OOV dictionary using validation and skip acoustics
    ${train_cmd} ${tempdir}/logs/validate_oov.log \
        mfa validate \
            -j ${nj} \
            --clean \
            --skip_acoustics \
            -t "${tempdir}" \
            ${single_spk_opt} \
            "${corpus_dir}" \
            "${dict_tag_or_path}" \
            --brackets ''

    # create new dictionary including OOV
    ${train_cmd} ${tempdir}/logs/generate_dict.log \
        mfa g2p \
            -j ${nj} \
            -t ${tempdir} \
            ${single_spk_opt} \
            "${tempdir}/corpus/oovs_found_${dictionary}.txt" \
            "${mfa_g2p}" \
            "${oov_dict}"

    cat "${src_dict}" "${oov_dict}" > "${workdir}/${dictionary}.dict"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Generating alignments using MFA model"

    if ${mfa_validate}; then
        # Validate data set with acoustics.
        log "Validating corpus..."
        ${train_cmd} ${tempdir}/logs/validate.log \
            mfa validate \
                -j ${nj} \
                --clean \
                -t "${tempdir}" \
                ${single_spk_opt} \
                "${corpus_dir}" \
                "${dict_tag_or_path}" \
                --acoustic_model_path "${acoustic_model}" \
                --brackets ''
    fi

    log "Obtaining alignments..."
    ${train_cmd} ${tempdir}/logs/align.log \
        mfa align -j ${nj} \
                --clean \
                -t "$(pwd)/${tempdir}" \
                --output_format "${textgrid_format}" \
                ${single_spk_opt} \
                "${corpus_dir}" \
                "${workdir}/${dictionary}.dict" \
                "${acoustic_model}" \
                "${workdir}/alignments"

    if ${clean_temp}; then
        log "WARNING: Removing all temp files..."
        rm -r ${tempdir}
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Prepare phoneme-text labels"

    python pyscripts/utils/mfa_format.py \
        validate \
        --corpus_dir "${corpus_dir}" \
        --textgrid_format "${textgrid_format}"

    python pyscripts/utils/mfa_format.py \
        durations \
        --samplerate "${samplerate}" \
        --hop_size "${hop_size}" \
        --corpus_dir "${corpus_dir}" \
        --textgrid_format "${textgrid_format}"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "stage 6: Prepare data sets with phoneme alignments"
    for dset in ${split_sets}; do
        utils/copy_data_dir.sh data/"${dset}"{,_phn}
        cp ${workdir}/text data/${dset}_phn/text
        utils/fix_data_dir.sh data/${dset}_phn

        utils/filter_scp.pl data/${dset}_phn/utt2spk ${workdir}/durations > data/${dset}_phn/durations
    done
fi

log "Successfully finished generating data and MFA alignments."
