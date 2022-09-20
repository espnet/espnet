#!/bin/bash

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



. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


db_root="downloads" 

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    bash -x local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # set filenames
    scp=data/train/wav.scp
    utt2spk=data/train/utt2spk
    spk2utt=data/train/spk2utt
    text=data/train/text

    # check file existence
    [ ! -e data/train ] && mkdir -p data/train
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}

    # make scp, utt2spk, and spk2utt
    find ${db_root}/qasr_tts-1.0/wavs -follow -name "*.wav" | sort | while read -r filename;do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${id} ${filename}" >> ${scp}
        echo "${id} qsr" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # make text usign the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    paste -d " " \
        <(cut -d "|" -f 1 < ${db_root}/qasr_tts-1.0/metadata.csv) \
        <(cut -d "|" -f 2 < ${db_root}/qasr_tts-1.0/metadata.csv) \
        > ${text}

    #bash -x utils/validate_data_dir.sh --no-feats data/train
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: utils/subset_data_dir.sg"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --last data/train 50 data/deveval
    utils/subset_data_dir.sh --last data/deveval 25 data/${eval_set}
    utils/subset_data_dir.sh --first data/deveval 25 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 50 ))
    utils/subset_data_dir.sh --first data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
