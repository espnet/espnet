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

if [ -z "${CLOTHO_ENTAILMENT}" ]; then
    log "Fill the value of 'CLOTHO_ENTAILMENT' of db.sh"
    exit 1
fi


log "stage 1: Data preparation"

DATADIR=$1
## DOWNLOAD DATA if CLOTHO_ENTAILMENT is set to downloads
if [ "${CLOTHO_ENTAILMENT}" == "downloads" ]; then
    # If there is no argument, the default download directory is set to currentdir/downloads
    CLOTHO_ENTAILMENT_ROOT_DIR="${DATADIR}/downloads"
    log "Downlaoding clotho into ${CLOTHO_ENTAILMENT_ROOT_DIR}."
    mkdir -p "${CLOTHO_ENTAILMENT_ROOT_DIR}"

    if [ ! -e "${CLOTHO_ENTAILMENT_ROOT_DIR}/download_done" ]; then
        log "stage 1: Data preparation - Installing aac-datasets"
        if ! pip3 install aac-datasets; then
            log "Error: Installing aac-datasets failed."
            exit 1
        fi
        for split in val eval dev; do
            log "Downloading ${split} split."
            if ! aac-datasets-download --root "${CLOTHO_ENTAILMENT_ROOT_DIR}" clotho --subsets "${split}"; then
                log "Error: Downloading Clotho dataset failed."
                exit 1
            fi
        done
        pwd_=$(pwd)
        cd "${CLOTHO_ENTAILMENT_ROOT_DIR}"
        git clone https://github.com/microsoft/AudioEntailment.git
        cd ${pwd_}
        touch "${CLOTHO_ENTAILMENT_ROOT_DIR}/download_done"
    else
        log "Clotho dataset is already downloaded. ${CLOTHO_ENTAILMENT_ROOT_DIR}/download_done exists."
    fi
else
    CLOTHO_ENTAILMENT_ROOT_DIR=${CLOTHO_ENTAILMENT}
fi

CLOTHO_ENTAILMENT_ROOT_DIR=$(realpath ${CLOTHO_ENTAILMENT_ROOT_DIR})
log "Using the data directory: ${CLOTHO_ENTAILMENT_ROOT_DIR}"

# Prepare data in the Kaldi format:
# text, hypothesis, wav.scp, utt2spk
# text is the label pos, neg, neutral
# hypothesis is an assertion about the audio premise 
##########
python3 local/data_prep_clotho_entailment.py ${CLOTHO_ENTAILMENT_ROOT_DIR} ${DATADIR}
##########

# SORT ALL
SPLITS=(development validation evaluation)
for split_name in "${SPLITS[@]}"; do
    for f in wav.scp utt2spk text hypothesis.txt; do
        if [ -f "${DATADIR}/${split_name}/${f}" ]; then
            sort "${DATADIR}/${split_name}/${f}" -o "${DATADIR}/${split_name}/${f}"
        fi
    done
    echo "Running spk2utt for ${split_name}"
    utils/utt2spk_to_spk2utt.pl "${DATADIR}/${split_name}/utt2spk" > "${DATADIR}/${split_name}/spk2utt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"
