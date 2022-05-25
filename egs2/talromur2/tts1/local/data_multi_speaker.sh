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

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${TALROMUR2}" ]; then
   log "Fill the value of 'TALROMUR' of db.sh"
   exit 1
fi
db_root=${TALROMUR2}

full_set=full
train_set=train
deveval_set=deveval
dev_set=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

# set filenames
scp=data/${full_set}/wav.scp
utt2spk=data/${full_set}/utt2spk
spk2utt=data/${full_set}/spk2utt
text=data/${full_set}/text
num_dev=250
num_eval=250

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"

    # check file existence
    [ ! -e data/${full_set} ] && mkdir -p data/${full_set}
    [ -e ${scp} ] && rm ${scp}
    [ -e ${utt2spk} ] && rm ${utt2spk}
    [ -e ${spk2utt} ] && rm ${spk2utt}
    [ -e ${text} ] && rm ${text}
    [ -e data/alignments ] && rm -r data/alignments

    # make scp, utt2spk, and spk2utt
    find ${db_root}/s* -follow -name "*.wav" | sort | while read -r filename;do
        spk_id=$(basename $(dirname $(dirname ${filename})))
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${spk_id}" >> ${utt2spk}
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
        #filepath is .../<spk_id>/<filename>
        FULLPATH=$(realpath ${filepath})
        spk_id=$(basename $(dirname ${filepath}))
        filename=$(basename $filepath)
        if [ ! -e data/alignments/$filename ]; then
            ln -s $FULLPATH data/alignments/$filename 
        fi
    done

    # Trim leading and trailing silences from audio using sox
    echo "Trimming audio"
    python local/data_utils.py data/alignments ${scp}

    # make text using the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    echo "Collecting text prompts"
    for path in downloads/s*
    do
        speaker_id=$(basename ${path})
        paste -d " " \
            <(cut -f 1 < ${db_root}/${speaker_id}/index.tsv) \
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

    num_all=$(wc -l < "${scp}")
    num_deveval=$((num_dev + num_eval))
    num_train=$((num_all - num_deveval))
    utils/subset_data_dir.sh --last "data/${full_set}" "${num_deveval}" "data/${deveval_set}"
    utils/subset_data_dir.sh --first "data/${deveval_set}" "${num_dev}" "data/${eval_set}"
    utils/subset_data_dir.sh --last "data/${deveval_set}" "${num_eval}" "data/${dev_set}"
    utils/subset_data_dir.sh --first "data/${full_set}" "${num_train}" "data/${train_set}"
    # # make evaluation and devlopment sets
    # for path in downloads/s*
    # do
    #     speaker_id=$(basename ${path})
    #     paste -d " " \
    #         <(cut -f 1 < ${db_root}/split/${speaker_id}_val.txt) \
    #         >> val_utts.txt
    #     paste -d " " \
    #         <(cut -f 1 < ${db_root}/split/${speaker_id}_test.txt) \
    #         >> test_utts.txt
    #     paste -d " " \
    #         <(cut -f 1 < ${db_root}/split/${speaker_id}_train.txt) \
    #         >> train_utts.txt
    # done
    # utils/subset_data_dir.sh --utt-list val_utts.txt data/${full_set} data/${dev_set}
    # utils/subset_data_dir.sh --utt-list test_utts.txt data/${full_set} data/${eval_set}
    # utils/subset_data_dir.sh --utt-list train_utts.txt data/${full_set} data/${train_set}

    # rm val_utts.txt test_utts.txt train_utts.txt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
