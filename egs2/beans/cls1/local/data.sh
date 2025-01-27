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

DATA_PREP_ROOT=${1:-"."}
DATASETS=${2:-"dcase"}

if [ -z "${BEANS}" ]; then
    log "Fill the value of 'BEANS' in db.sh"
    exit 1
fi


# DCASE
if [[ "${DATASETS}" == *"dcase"* ]]; then
    log "Processing DCASE 21"
    DCASE_LOCATION="${DATA_PREP_ROOT}/downloads/dcase"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/dcase/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            mkdir -p ${DATA_PREP_ROOT}/{wav,downloads}
            wget -O ${DATA_PREP_ROOT}/downloads/Development_Set.zip https://zenodo.org/record/5412896/files/Development_Set.zip?download=1
            unzip ${DATA_PREP_ROOT}/downloads/Development_Set.zip -d ${DCASE_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/Development_Set.zip
            touch "${DCASE_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/dcase"
        DCASE_LOCATION="${BEANS}/dcase"
    fi
    python local/scripts/dcase.py ${DCASE_LOCATION} ${DATA_PREP_ROOT}
fi


for dataset in ${DATASETS}; do
    for x in ${dataset}.dev ${dataset}.train ${dataset}.test; do
        for f in text wav.scp utt2spk; do
            if [ -f "${DATA_PREP_ROOT}/${x}/${f}" ]; then
                sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
            fi
        done
        if [ -f "${DATA_PREP_ROOT}/${x}/utt2spk" ]; then
            utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
            utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
        fi
    done
done


log "Successfully finished. [elapsed=${SECONDS}s]"