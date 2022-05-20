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
stop_stage=2
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SLURP}" ]; then
    log "Fill the value of 'SLURP' of db.sh"
    exit 1
fi

if [ -z "${SLURP_S}" ]; then
    log "Fill the value of 'SLURP_S' of db.sh"
    exit 1
fi


if [ -z "${LIBRITRANS_S}" ]; then
    log "Fill the value of 'LIBRITRANS_S' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${SLURP}/LICENSE.txt" ]; then
    	log "Data Preparation stage 1: Download data to ${SLURP}"
        git clone https://github.com/pswietojanski/slurp.git ${SLURP}
    else
        log "Data Preparation stage 1: ${SLURP}/LICENSE.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Data Preparation stage 2: Mixture Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/prepare_mixture_data.py ${SLURP} ${SLURP_S} ${LIBRITRANS_S}
    for x in lt_test lt_test_qut slurp_test slurp_test_qut devel train; do
        for f in text wav.scp utt2spk spk1.scp; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
