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

stage=1
stop_stage=100000
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

DATA_PREP_ROOT=${1:-"."}
FOLD=${2:-1}

if [ -z "${ESC50}" ]; then
    log "Fill the value of 'ESC50' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${ESC50}/LICENSE" ]; then
	    echo "stage 1: Download data to ${ESC50}"
    else
        log "stage 1: ${ESC50}/LICENSE is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    # Prepare data for 5-fold cross-validation
    echo "Preparing data for fold ${FOLD}"
    python3 local/data_prep_esc50.py ${ESC50} ${FOLD} ${DATA_PREP_ROOT}
    for x in val${FOLD} train${FOLD}; do
        for f in text wav.scp utt2spk; do
            sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
