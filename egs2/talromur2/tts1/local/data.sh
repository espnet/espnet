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
dev_set=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    if [ ${db_root} == "downloads" ]; then
        log "Non-default db root provided, skipping download..."
    else
        ./local/data_download.sh "${db_root}"
    fi
fi

# set filenames
scp=data/${full_set}/wav.scp
utt2spk=data/${full_set}/utt2spk
spk2utt=data/${full_set}/spk2utt
text=data/${full_set}/text

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
        spk_dir=$(dirname ${filename})
        spk_id=$(basename "$(dirname ${spk_dir})")
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${spk_id}" >> ${utt2spk}
    done
    sort ${utt2spk} > ${utt2spk}_sorted
    mv ${utt2spk}_sorted ${utt2spk}
    sort ${scp} > ${scp}_sorted
    mv ${scp}_sorted ${scp}
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    if [ -e ${db_root}/alignments ]
    then
        # Collect all alignment files in one directory
        echo "Gathering alignment data"
        mkdir -p data/alignments
        find ${db_root}/alignments/ -name "*.TextGrid" -type f | while read -r filepath;do
            #filepath is .../<spk_id>/<filename>
            FULLPATH=$(realpath ${filepath})
            spk_id=$(basename "$(dirname ${filepath})")
            filename=$(basename $filepath)
            if [ ! -e data/alignments/$filename ]; then
                ln -s $FULLPATH data/alignments/$filename
            fi
        done

        # Trim leading and trailing silences from audio using sox
        echo "Trimming audio"
        python local/data_utils.py data/alignments ${scp}
    fi

    # make text using the original text
    # cleaning and phoneme conversion are performed on-the-fly during the training
    echo "Collecting text prompts"
    for path in "${db_root}"/s*
    do
        speaker_id=$(basename ${path})
        paste -d " " \
            <(cut -f 1 < ${db_root}/${speaker_id}/index.tsv) \
            <(cut -f 3 < ${db_root}/${speaker_id}/index.tsv) \
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

    ./local/split_train_dev_test.py \
        --data_dir "data/${full_set}" \
        --train_dir "data/${train_set}" \
        --dev_dir "data/${dev_set}" \
        --test_dir "data/${eval_set}"


fi

log "Successfully finished. [elapsed=${SECONDS}s]"
