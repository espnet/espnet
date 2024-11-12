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

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${CLOTHO_V2}" ]; then
    log "Fill the value of 'CLOTHO_V2' of db.sh"
    exit 1
fi

log "stage 1: Data preparation"

## DOWNLOAD DATA
# If there is no argument, the default download directory is set to currentdir/downloads
if [ $# -ne 1 ]; then
    CLOTHO_V2_ROOT_DIR="$(pwd)/downloads"
    log "Using the default download directory: ${CLOTHO_V2_ROOT_DIR}"
else
    CLOTHO_V2_ROOT_DIR="$1/downloads"
    log "Using the specified download directory: ${CLOTHO_V2_ROOT_DIR}"
fi


if [ ! -e "${CLOTHO_V2_ROOT_DIR}/download_done" ]; then
    log "stage 1: Data preparation - Installing aac-datasets"
    if ! pip3 install aac-datasets; then
        echo "Error: Installing aac-datasets failed."
        exit 1
    fi
    echo "Downlaoding clotho into ${CLOTHO_V2_ROOT_DIR}."
    mkdir -p "${CLOTHO_V2_ROOT_DIR}"
    for split in val eval dev; do
        echo "Downloading ${split} split."
        if ! aac-datasets-download --root "${CLOTHO_V2_ROOT_DIR}" clotho --subsets "${split}"; then
            echo "Error: Downloading Clotho dataset failed."
            exit 1
        fi
    done
    touch "${CLOTHO_V2_ROOT_DIR}/download_done"
else
    echo "Clotho dataset is already downloaded. ${CLOTHO_V2_ROOT_DIR}/download_done exists."
fi

CLOTHO_V2_ROOT_DIR="${CLOTHO_V2_ROOT_DIR}/CLOTHO_v2.1"

## PREPARE DATA
if [ ! -d ${CLOTHO_V2_ROOT_DIR} ]; then
    echo Cannot find ${CLOTHO_V2_ROOT_DIR} directory! Exiting...
    exit 1
fi
if [ -z "${AUDIOCAPS}" ]; then
    log "Fill the value of 'AUDIOCAPS' of db.sh"
    exit 1
fi
if [ -z "${CLOTHO_CHATGPT_MIXUP}" ]; then
    log "Fill the value of 'CLOTHO_CHATGPT_MIXUP' of db.sh"
    exit 1
fi

AUDIOCAPS_ROOT_DIR=${AUDIOCAPS}
if [ ! -d ${AUDIOCAPS_ROOT_DIR} ]; then
    echo Cannot find ${AUDIOCAPS_ROOT_DIR} directory! Exiting...
    exit 1
fi
CLOTHO_CHATGPT_MIXUP_ROOT_DIR=${CLOTHO_CHATGPT_MIXUP}
if [ ! -d ${CLOTHO_CHATGPT_MIXUP} ]; then
    echo Cannot find ${CLOTHO_CHATGPT_MIXUP} directory! Exiting...
    exit 1
fi

SPLITS=(development_clotho validation evaluation) # map dev, val, eval to development, validation, evaluation
N_REF=5
for split_name in ${SPLITS[@]}; do
    mkdir -p "data/${split_name}"
done

echo "$(which python)"
# Prepare data in the Kaldi format, including three files:
# text, wav.scp, utt2spk
##########
python3 local/data_prep_clotho.py ${CLOTHO_V2_ROOT_DIR} ${N_REF}
##########
mkdir -p data/development_audiocaps
python3 local/data_prep_audiocaps.py ${AUDIOCAPS_ROOT_DIR}
SPLITS+=(development_audiocaps)
##########
mkdir -p data/development_clotho_chatgpt_mixup
python3 local/data_prep_clotho_chatgpt_mixup.py ${CLOTHO_CHATGPT_MIXUP_ROOT_DIR} ${CLOTHO_V2_ROOT_DIR}
SPLITS+=(development_clotho_chatgpt_mixup)
##########

# CONCATENATE MIXUP AND AUDIOCAPS
mkdir -p data/pretrain
for f in wav.scp utt2spk text; do
    cat data/development_audiocaps/${f} data/development_clotho_chatgpt_mixup/${f} > data/pretrain/${f}
done
SPLITS+=(pretrain)

# SIX directories: 4 training sets - development_clotho, development_audiocaps, development_clotho_chatgpt_mixup, development_pretrain
# and 2 for val, eval - validation, evaluation

# SORT ALL
for split_name in ${SPLITS[@]}; do
    for f in wav.scp utt2spk; do
        sort data/${split_name}/${f} -o data/${split_name}/${f}
    done
    # Sort all text files
    if [ -f data/${split_name}/text ]; then
        sort data/${split_name}/text -o data/${split_name}/text
    fi
    for i in $(seq ${N_REF}); do
        if [ -f data/${split_name}/text_spk${i} ]; then
            sort data/${split_name}/text_spk${i} -o data/${split_name}/text_spk${i}
        fi
    done
    echo "Running spk2utt for ${split_name}"
    utils/utt2spk_to_spk2utt.pl data/${split_name}/utt2spk > "data/${split_name}/spk2utt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"
