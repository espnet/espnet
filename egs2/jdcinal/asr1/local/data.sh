#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${JDCINAL}" ]; then
    log "Fill the value of 'JDCINAL' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Download data to ${JDCINAL}"
    if [ ! -d "${JDCINAL}" ]; then
        mkdir -p "${JDCINAL}"
    fi
    url=http://tts.speech.cs.cmu.edu/awb/infomation_navigation_and_attentive_listening_0.2.zip
    wget ${url} ${JDCINAL}

    log "Unzip data to ${JDCINAL}"
    unzip -q ${JDCINAL}infomation_navigation_and_attentive_listening_0.2.zip -d ${JDCINAL}

    # Data Preparation
    mkdir -p data/{train,valid,test}
    mkdir -p data/tmp
    echo -n "" > data/tmp/wav.scp; echo -n "" > data/tmp/utt2spk; echo -n "" > data/tmp/text; echo -n "" > data/tmp/segments
    for m in C P S; do
        for n in {1..4}; do
            for file in "${JDCINAL}"/infomation_navigation_and_attentive_listening_0.2/"${m}""${n}"/*.csv; do
                dos2unix -q $file
                ses_id=$(basename "${file}" .csv)
                wav_file="${JDCINAL}"/infomation_navigation_and_attentive_listening_0.2/sound/${ses_id}.trim.wav
                #create files
                iconv -f SJIS -t UTF8 ${file} > data/tmp/${ses_id}
                perl local/csv2file.pl data/tmp/${ses_id} ${wav_file}
                rm data/tmp/${ses_id}
            done
        done
    done
    dos2unix -q data/tmp/wav.scp; dos2unix -q data/tmp/utt2spk; dos2unix -q data/tmp/text; dos2unix -q data/tmp/segments
    sort -u data/tmp/wav.scp -o data/tmp/wav.scp
    sort -u data/tmp/utt2spk -o data/tmp/utt2spk
    sort -u data/tmp/text -o data/tmp/text
    sort -u data/tmp/segments -o data/tmp/segments
    utils/utt2spk_to_spk2utt.pl data/tmp/utt2spk > "data/tmp/spk2utt"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # make train valid test sets
    for file in wav.scp utt2spk spk2utt text segments; do
        grep -e "C1-" -e "C2-" -e "C3-" -e "C4-" -e "P1-" -e "P2-" -e "P3-" data/tmp/${file} > data/train/${file}
        grep "P4-" data/tmp/${file} > data/valid/${file}
        grep -e "S1-" -e "S2-" -e "S3-" -e "S4-" data/tmp/${file} > data/test/${file}
    done
    for dset in test valid train; do
        utils/validate_data_dir.sh --no-feats --non-print data/${dset} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
