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

. ./db.sh
. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CATSLU}" ]; then
    log "Fill the value of 'CATSLU' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${CATSLU}/catslu_traindev" ] && [ ! -e "${CATSLU}/catslu_test" ]; then
	echo "stage 1: Download traindev and test data to ${CATSLU}"
    else
        log "stage 1: ${CATSLU}/catslu_traindev and ${CATSLU}/catslu_test already exists."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/data_prep.py ${CATSLU}
    for x in test valid train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
    done
    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/valid
    utils/fix_data_dir.sh data/test

    utils/validate_data_dir.sh --no-feats data/train
    utils/validate_data_dir.sh --no-feats data/valid
    utils/validate_data_dir.sh --no-feats data/test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
