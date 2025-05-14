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
stop_stage=100
data_url=https://huggingface.co/datasets/chenehk/allsstar_eng/resolve/main

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${ALLSSTAR_ENG}" ]; then
    log "Fill the value of 'ALLSSTAR_ENG' of db.sh"
    exit 1
fi

datadir=${ALLSSTAR_ENG}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    mkdir -p ${datadir}
    if ! local/download_and_untar.sh ${datadir} ${data_url}; then
        log "Failed to download"
        exit 1
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    if [ ! -f ${datadir}/metadata.csv ]; then
        echo Cannot find allsstar root! Exiting...
        exit 1
    fi

    for x in test_l1 test_l2; do
        opt=""
        [ "${x}" == "test_l2" ] && opt="-v"

        mkdir -p data/${x}
        tail -n+2 ${datadir}/metadata.csv | grep $opt _ENG_ENG_ > data/${x}/metadata.csv

        cut -d, -f1,2 data/${x}/metadata.csv | sed "s|,| ${datadir}/|" | sort > data/${x}/wav.scp
        cut -d, -f1,3 data/${x}/metadata.csv | sed "s|,| |" | sort > data/${x}/utt2spk
        cut -d, -f1,4- data/${x}/metadata.csv | sed "s|,| |" | sort > data/${x}/text
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
