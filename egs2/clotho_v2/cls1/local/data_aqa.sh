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

if [ -z "${CLOTHO_AQA}" ]; then
    log "Fill the value of 'CLOTHO_AQA' of db.sh"
    exit 1
fi


log "stage 1: Data preparation"

DATADIR=$1
TYPE=$2
## DOWNLOAD DATA if CLOTHO_AQA is set to downloads
if [ "${CLOTHO_AQA}" == "downloads" ]; then
    # If there is no argument, the default download directory is set to currentdir/downloads
    CLOTHO_AQA_ROOT_DIR="${DATADIR}/downloads/AQA"
    log "Downlaoding clotho into ${CLOTHO_AQA_ROOT_DIR}."
    mkdir -p "${CLOTHO_AQA_ROOT_DIR}"

    if [ ! -e "${CLOTHO_AQA_ROOT_DIR}/aqa_download_done" ]; then
        if [ ! -e "${CLOTHO_AQA_ROOT_DIR}/audio_download_done" ]; then    
            log "stage 1: Data preparation"
            wget -P ${CLOTHO_AQA_ROOT_DIR}/ https://zenodo.org/records/6473207/files/audio_files.zip
            unzip ${CLOTHO_AQA_ROOT_DIR}/audio_files.zip -d ${CLOTHO_AQA_ROOT_DIR}
            touch "${CLOTHO_AQA_ROOT_DIR}/audio_download_done"
        else
            log "Clotho audio is already downloaded. ${CLOTHO_AQA_ROOT_DIR}/audio_download_done exists."
        fi
        for split in test val train; do
            wget -P ${CLOTHO_AQA_ROOT_DIR}/ "https://zenodo.org/records/6473207/files/clotho_aqa_${split}.csv"
        done        
        touch "${CLOTHO_AQA_ROOT_DIR}/aqa_download_done"
    else
        log "Clotho AQA dataset is already downloaded. ${CLOTHO_AQA_ROOT_DIR}/aqa_download_done exists."
    fi
else
    CLOTHO_AQA_ROOT_DIR=${CLOTHO_AQA}
fi

CLOTHO_AQA_ROOT_DIR=$(realpath ${CLOTHO_AQA_ROOT_DIR})
log "Using the data directory: ${CLOTHO_AQA_ROOT_DIR}"

# Prepare data in the Kaldi format:
# text, question.txt, wav.scp, utt2spk
# text is the label pos, neg, neutral
# question.txt contains question about the audio
##########
python3 local/data_prep_clotho_aqa.py ${CLOTHO_AQA_ROOT_DIR} ${DATADIR} ${TYPE}
##########

# SORT ALL
SPLITS=(development_aqa_${TYPE} validation_aqa_${TYPE} evaluation_aqa_${TYPE})
for split_name in "${SPLITS[@]}"; do
    for f in wav.scp utt2spk text question.txt; do
        if [ -f "${DATADIR}/${split_name}/${f}" ]; then
            sort "${DATADIR}/${split_name}/${f}" -o "${DATADIR}/${split_name}/${f}"
        fi
    done
    echo "Running spk2utt for ${split_name}"
    utils/utt2spk_to_spk2utt.pl "${DATADIR}/${split_name}/utt2spk" > "${DATADIR}/${split_name}/spk2utt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"
