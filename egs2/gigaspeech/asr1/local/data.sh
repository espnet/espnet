#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=5000
data_dir="data"

log "$0 $*"
. utils/parse_options.sh

# base url for downloads.
giga_repo=https://github.com/SpeechColab/GigaSpeech.git

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${GIGASPEECH}" ]; then
    log "Fill the value of 'GIGASPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -d "${GIGASPEECH}/audio" ] && [ -f "${GIGASPEECH}/GigaSpeech.json" ]; then
	log "GIGASPEECH found in ${GIGASPEECH}."
	rm -fr GigaSpeech
	git clone $giga_repo
    else
	echo "Valid GIGASPEECH data not found in ${GIGASPEECH}."
	echo "Please follow the instruction in https://github.com/SpeechColab/GigaSpeech#dataset-download"
	echo "and re-construct the data."
	exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "data preparation"
    mkdir -p ${data_dir}
    abs_data_dir=$(readlink -f ${data_dir})
    log "making Kaldi format data directory in ${abs_data_dir}"
    pushd GigaSpeech
    ./toolkits/kaldi/gigaspeech_data_prep.sh --train-subset XL ${GIGASPEECH} ${abs_data_dir}
    popd
    mv ${data_dir}/gigaspeech_train_xl ${data_dir}/train
    mv ${data_dir}/gigaspeech_dev ${data_dir}/dev
    mv ${data_dir}/gigaspeech_test ${data_dir}/test
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "fixing data directories"
    for sub in train dev test; do
	utils/fix_data_dir.sh ${data_dir}/${sub}
	# reco2dur causes the error at stage 4 in asr.sh
	rm -f ${data_dir}/${sub}/reco2dur
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
