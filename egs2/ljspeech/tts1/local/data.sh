#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${LJSPEECH}" ]; then
   log "Fill the value of 'LJSPEECH' of db.sh"
   exit 1
fi
db_root=${LJSPEECH}

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # set filenames
    scp=data/train/wav.scp
    utt2spk=data/train/utt2spk
    spk2utt=data/train/spk2utt
    text=data/train/text
    durations=data/train/durations

    # check file existence
    [ ! -e data/train ] && mkdir -p data/train
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}
    [ -e ${durations} ] && rm ${durations}

    wavs_dir="${db_root}/LJSpeech-1.1/wavs"
    # make scp, utt2spk, and spk2utt
    find "${wavs_dir}" -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${id} ${filename}" >> ${scp}
        echo "${id} LJ" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # make text using the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    paste -d " " \
        <(cut -d "|" -f 1 < ${db_root}/LJSpeech-1.1/metadata.csv) \
        <(cut -d "|" -f 3 < ${db_root}/LJSpeech-1.1/metadata.csv) \
        > ${text}

    utils/validate_data_dir.sh --no-feats data/train
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --last data/train 500 data/deveval
    utils/subset_data_dir.sh --last data/deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/deveval 250 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
