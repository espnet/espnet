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

log "$0 $*"
. utils/parse_options.sh

# base url for downloads.
giga_repo=https://github.com/SpeechColab/GigaSpeech.git

# dirs
data_dir=data
meta_dir=data/meta

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
    log "creating meta data in ${meta_dir}"
    python3 GigaSpeech/utils/analyze_meta.py --pipe-format ${GIGASPEECH}/GigaSpeech.json ${meta_dir}
fi

declare -A dic_sets
dic_sets=([train]="XL" [dev]="DEV" [test]="TEST")
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "making Kaldi data structure (all data)"
    # all corpus
    [ ! -d ${data_dir}/corpus ] && mkdir -p ${data_dir}/corpus

    for f in utt2spk wav.scp text segments utt2dur reco2dur; do
	[ -f ${meta_dir}/$f ] && cp ${meta_dir}/${f} ${data_dir}/corpus/
    done

    utt2spk=${data_dir}/corpus/utt2spk
    spk2utt=${data_dir}/corpus/spk2utt
    utt2spk_to_spk2utt.pl <${utt2spk} >${spk2utt} || (echo "Error: utt2spk to spk2utt" && exit 1)

    # Delete <*> tag
    sed -i '/<MUSIC>/d' ${data_dir}/corpus/text
    sed -i '/<NOISE>/d' ${data_dir}/corpus/text
    sed -i "s|<[^>]*>||g" ${data_dir}/corpus/text
    sed -i 's/[ ][ ]*/ /g' ${data_dir}/corpus/text
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "splitting train, dev, and test directories"
    # train dev test
    [ ! -f ${meta_dir}/utt2subsets ] && echo "Error: No such file ${meta_dir}/utt2subsets!" && exit 1
    for sub in ${!dic_sets[*]}; do
	[ ! -d ${data_dir}/${sub} ] && mkdir -p ${data_dir}/${sub}
	tag=${dic_sets[${sub}]}
	grep "{$tag}" ${meta_dir}/utt2subsets | subset_data_dir.sh --utt-list - ${data_dir}/corpus ${data_dir}/${sub}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
