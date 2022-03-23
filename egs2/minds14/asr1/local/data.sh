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

language=$1

if [ $# -ne 1 ]; then
    log "Error: pass in [language-LOCALE] as argument."
    exit 2
fi

if [ -z "${MINDS14_DIR}" ]; then
    log "Fill the value of 'MINDS14_DIR' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${MINDS14_DIR}/license.md" ]; then
	echo "stage 1: Download data to ${MINDS14_DIR}"
    wget https://poly-public-data.s3.amazonaws.com/MInDS-14/MInDS-14.zip
    unzip MInDS-14.zip
    unzip ${MINDS14_DIR}/audio.zip -d ${MINDS14_DIR}/audio
    unzip ${MINDS14_DIR}/text.zip -d ${MINDS14_DIR}/text
    else
        log "stage 1: ${MINDS14_DIR}/LICENCE already exists. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/data_prep.py ${MINDS14_DIR} ${language}
    for x in test valid train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
