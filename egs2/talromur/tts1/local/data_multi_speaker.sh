#!/usr/bin/env bash

# Copyright 2021 Gunnar Thor Örnólfsson
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

. utils/parse_options.sh

# if [ $# -ne 1 ]; then
#     log "Error: Only speaker id is required. It should be one of the values [a, b, c, d, e, f, g, h]"
#     exit 2
# fi

# speaker_id=$1

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${TALROMUR}" ]; then
   log "Fill the value of 'TALROMUR' of db.sh"
   exit 1
fi
db_root=${TALROMUR}

full_set=full_all
train_set=train_all
deveval_set=deveval_all
train_dev=dev_all
eval_set=eval1_all

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # set filenames
    scp=data/${full_set}/wav.scp
    utt2spk=data/${full_set}/utt2spk
    spk2utt=data/${full_set}/spk2utt
    text=data/${full_set}/text

    # check file existence
    [ ! -e data/${full_set} ] && mkdir -p data/${full_set}
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}

    # make scp, utt2spk, and spk2utt
    find ${db_root}/{a,b,c,d,e,f,g,h} -follow -name "*.wav" | sort | while read -r filename;do
        spk_id=$(basename $(dirname $(dirname ${filename})))
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${spk_id}_${id} ${filename}" >> ${scp}
        echo "${spk_id}_${id} ${spk_id}" >> ${utt2spk}
    done
    sort ${utt2spk} > ${utt2spk}_sorted
    mv ${utt2spk}_sorted ${utt2spk}
    sort ${scp} > ${scp}_sorted
    mv ${scp}_sorted ${scp}
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    # Collect all alignment files in one directory
    echo "Gathering alignment data"
    mkdir -p data/alignments
    find ${db_root}/alignments/ -name "*.TextGrid" -type f | while read -r filepath;do
        #filepath is .../<spk_id>/audio/<filename>
        spk_id=$(basename $(dirname $(dirname ${filepath})))
        filename=$(basename $filepath)
        if [ ! -e data/alignments/${spk_id}_$filename ]; then
            ln -s $filepath data/alignments/${spk_id}_$filename 
        fi
    done

    # Trim leading and trailing silences from audio using sox
    echo "Trimming audio"
    python local/data_utils.py data/alignments ${scp}

    # make text usign the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    echo "Collecting text prompts"
    for speaker_id in "a" "b" "c" "d" "e" "f" "g" "h"
    do
        paste -d " " \
            <(cut -f 1 < ${db_root}/${speaker_id}/index.tsv | sed "s/^/${speaker_id}_/") \
            <(cut -f 4 < ${db_root}/${speaker_id}/index.tsv) \
            >> ${text}
    done
    sort ${text} > ${text}_sorted
    mv ${text}_sorted ${text}
    echo "Validating data directory"
    utils/fix_data_dir.sh data/${full_set}
    utils/validate_data_dir.sh --no-feats data/${full_set}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    if [ -e val_utts.txt ]; then rm val_utts.txt; fi
    if [ -e test_utts.txt ]; then rm test_utts.txt; fi
    if [ -e train_utts.txt ]; then rm train_utts.txt; fi
    # make evaluation and devlopment sets
    for speaker_id in "a" "b" "c" "d" "e" "f" "g" "h"
    do
        paste -d " " \
            <(cut -f 1 < ${db_root}/split/${speaker_id}_val.txt | sed "s/^/${speaker_id}_/") \
            >> val_utts.txt
        paste -d " " \
            <(cut -f 1 < ${db_root}/split/${speaker_id}_test.txt | sed "s/^/${speaker_id}_/") \
            >> test_utts.txt
        paste -d " " \
            <(cut -f 1 < ${db_root}/split/${speaker_id}_train.txt | sed "s/^/${speaker_id}_/") \
            >> train_utts.txt
    done
    utils/subset_data_dir.sh --utt-list val_utts.txt data/${full_set} data/${train_dev}
    utils/subset_data_dir.sh --utt-list test_utts.txt data/${full_set} data/${eval_set}
    utils/subset_data_dir.sh --utt-list train_utts.txt data/${full_set} data/${train_set}

    rm val_utts.txt test_utts.txt train_utts.txt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"