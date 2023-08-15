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

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${AR_SC}" ]; then
    log "Fill the value of 'AR_SC' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${AR_SC}/README.md" ]; then
	    echo "stage 1: Download data to ${AR_SC}"
        git clone https://github.com/ltkbenamer/AR_Speech_Database.git ${AR_SC}
    else
        log "stage 1: ${AR_SC}/raw/words is already existing. Skip data downloading"
    fi

    cur_dir=$(pwd)
    cd ${AR_SC}
    if [ ! -d ${AR_SC}/SpeechAdvReprogram ]; then
        git clone https://github.com/dodohow1011/SpeechAdvReprogram.git
    fi
    cd ${cur_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/data_prep.py ${AR_SC}
    for x in test valid train; do
        for f in text wav.scp; do
            echo ${x}, ${f}
            echo "?"
            head data/${x}/${f}
            sort data/${x}/${f} -o data/${x}/${f}
            echo "====="
            head data/${x}/${f}
        done
        #utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats --no-text data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
