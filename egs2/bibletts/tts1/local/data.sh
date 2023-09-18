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
lang="hausa"
subsets=("dev" "eval" "train") # 
log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${BIBLETTS}" ]; then
   log "Fill the value of 'BIBLETTS' of db.sh"
   exit 1
fi
db_root=${BIBLETTS}

train_set=${lang}/tr_no_dev
train_dev=${lang}/dev1
eval_set=${lang}/eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for subset in "${subsets[@]}"; do
        log "stage 0: Data Preparation for subset $subset"
        # set filenames
        scp=data/${subset}/wav.scp
        utt2spk=data/${subset}/utt2spk
        spk2utt=data/${subset}/spk2utt
        text=data/${subset}/text
        durations=data/${subset}/durations

        # check file existence
        [ ! -e data/${subset} ] && mkdir -p data/${subset}
        [ -e ${scp} ] && rm ${scp}
        [ -e ${utt2spk} ] && rm ${utt2spk}
        [ -e ${spk2utt} ] && rm ${spk2utt}
        [ -e ${text} ] && rm ${text}
        [ -e ${durations} ] && rm ${durations}

        wavs_dir="${db_root}/${lang}"
        # make scp, utt2spk, and spk2utt
        find "${wavs_dir}" -name "*.flac" | sort | while read -r filename; do
            id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
            echo "${id} ${filename}" >> ${scp}
            echo "${id} Bible" >> ${utt2spk}
        done
        utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

        # make text using the original text
        # cleaning and phoneme conversion are performed on-the-fly during the training
        paste -d " " \
            <(cut -d "|" -f 1 < ${db_root}/${lang}/text/${subset}.csv) \
            <(cut -d "|" -f 2 < ${db_root}/${lang}/text/${subset}.csv) \
            > ${text}
        utils/fix_data_dir.sh data/${subset}
        utils/validate_data_dir.sh --no-feats data/${subset}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    dev_num=$(wc -l < data/dev/wav.scp)
    eval_num=$(wc -l < data/eval/wav.scp)
    train_num=$(wc -l < data/train/wav.scp)
    echo $dev_num
    utils/subset_data_dir.sh --first data/train ${train_num} data/${train_set}
    utils/subset_data_dir.sh --first data/eval ${eval_num} data/${eval_set}
    utils/subset_data_dir.sh --first data/dev ${dev_num} data/${train_dev}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
