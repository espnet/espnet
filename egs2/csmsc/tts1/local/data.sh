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

if [ -z "${CSMSC}" ]; then
   log "Fill the value of 'CSMSC' of db.sh"
   exit 1
fi
db_root=${CSMSC}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"

    # check directory existence
    [ ! -e data/train ] && mkdir -p data/train

    # set filenames
    scp=data/train/wav.scp
    utt2spk=data/train/utt2spk
    spk2utt=data/train/spk2utt
    text=data/train/text
    segments=data/train/segments

    # check file existence
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${text} ] && rm ${text}
    [ -e ${segments} ] && rm ${segments}

    # make scp, utt2spk, and spk2utt
    find ${db_root}/CSMSC/Wave -name "*.wav" -follow | sort | while read -r filename; do
        id="$(basename ${filename} .wav)"
        echo "${id} ${filename}" >> ${scp}
        echo "${id} csmsc" >> ${utt2spk}
    done
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # make text
    # TODO(kan-bayashi): Use the text cleaner during training
    nkf -Lu -w ${db_root}/CSMSC/ProsodyLabeling/000001-010000.txt \
        | grep ^0 \
        | sed -e "s/\#[1-4]//g" -e "s/：/，/g" -e 's/“//g' -e 's/”//g' \
        | sed -e "s/（//g" -e "s/）//g"  -e "s/；/，/g" -e 's/…。/。/g' \
        | sed -e 's/——/，/g' -e 's/……/，/g' -e "s/、/，/g" \
        | sed -e 's/…/，/g' -e 's/—/，/g' > ${text}
    # Before: 今天#3，快递员#1拿着#1一个#1快递#1在#1办公室#1喊#3：秦王#1是#1哪个#3，有他#1快递#4？
    # After: 今天，快递员拿着一个快递在办公室喊，秦王是哪个，有他快递？

    # make segmente
    find ${db_root}/CSMSC/PhoneLabeling -name "*.interval" -follow | sort | while read -r filename; do
        id="$(basename ${filename} .interval)"
        start_sec=$(nkf -Lu -w ${filename} | tail -n +14 | head -n 1)
        end_sec=$(nkf -Lu -w ${filename} | head -n -2 | tail -n 1)
        echo "${id} ${id} ${start_sec} ${end_sec}" >> ${segments}
    done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --last data/train 200 data/deveval
    utils/subset_data_dir.sh --last data/deveval 100 data/${recog_set}
    utils/subset_data_dir.sh --first data/deveval 100 data/${train_dev}
    n=$(( $(wc -l < data/train/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
