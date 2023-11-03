#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=5
num_eval=5
train_set="tr_no_dev"
dev_set="dev"
eval_set="eval1"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 [Options] <db>"
    echo "e.g.: $0 downloads/VCTK-Corpus"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=${num_dev})."
    echo "    --num_eval: number of evaluation uttreances (default=${num_eval})."
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    exit 1
fi

set -euo pipefail

# NOTE(kan-bayashi): p315 will not be used since it lacks txt data
spks=$(find "${db}/wav48" -maxdepth 1 -name "p*" -exec basename {} \; | sort | grep -v p315)
train_data_dirs=""
dev_data_dirs=""
eval_data_dirs=""
for spk in ${spks}; do
    # check spk existence
    [ ! -e "${db}/lab/mono/${spk}" ] && \
        echo "${spk} does not exist." >&2 && exit 1;

    [ ! -e data/${spk}_train ] && mkdir -p data/${spk}_train

    # set filenames
    scp=data/${spk}_train/wav.scp
    utt2spk=data/${spk}_train/utt2spk
    text=data/${spk}_train/text
    segments=data/${spk}_train/segments
    spk2utt=data/${spk}_train/spk2utt

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"
    [ -e "${segments}" ] && rm "${segments}"

    # make scp, text, and segments
    find "${db}/wav48/${spk}" -follow -name "*.wav" | sort | while read -r wav; do
        id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
        lab=${db}/lab/mono/${spk}/${id}.lab
        txt=${db}/txt/${spk}/${id}.txt

        # check lab existence
        if [ ! -e "${lab}" ]; then
            echo "${id} does not have a label file. skipped."
            continue
        fi
        if [ ! -e "${txt}" ]; then
            echo "${id} does not have a text file. skipped."
            continue
        fi

        echo "${id} ${wav}" >> "${scp}"
        echo "${id} ${spk}" >> "${utt2spk}"
        echo "${id} $(cat ${txt})" >> "${text}"

        utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

        # parse start and end time from HTS-style mono label
        idx=1
        while true; do
            next_idx=$((idx+1))
            next_symbol=$(sed -n "${next_idx}p" "${lab}" | awk '{print $3}')
            if [ "${next_symbol}" != "pau" ]; then
                start_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $2}')
                break
            fi
            idx=${next_idx}
        done
        idx=$(wc -l < "${lab}")
        while true; do
            prev_idx=$((idx-1))
            prev_symbol=$(sed -n "${prev_idx}p" "${lab}" | awk '{print $3}')
            if [ "${prev_symbol}" != "pau" ]; then
                end_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $1}')
                break
            fi
            idx=${prev_idx}
        done
        start_sec=$(echo "${start_nsec}*0.0000001" | bc | sed "s/^\./0./")
        end_sec=$(echo "${end_nsec}*0.0000001" | bc | sed "s/^\./0./")
        echo "${id} ${id} ${start_sec} ${end_sec}" >> "${segments}"
    done

    # split
    num_all=$(wc -l < "${scp}")
    num_deveval=$((num_dev + num_eval))
    num_train=$((num_all - num_deveval))
    utils/subset_data_dir.sh --last "data/${spk}_train" "${num_deveval}" "data/${spk}_deveval"
    utils/subset_data_dir.sh --first "data/${spk}_deveval" "${num_dev}" "data/${spk}_${eval_set}"
    utils/subset_data_dir.sh --last "data/${spk}_deveval" "${num_eval}" "data/${spk}_${dev_set}"
    utils/subset_data_dir.sh --first "data/${spk}_train" "${num_train}" "data/${spk}_${train_set}"

    # remove tmp directories
    rm -rf "data/${spk}_train"
    rm -rf "data/${spk}_deveval"

    train_data_dirs+=" data/${spk}_${train_set}"
    dev_data_dirs+=" data/${spk}_${dev_set}"
    eval_data_dirs+=" data/${spk}_${eval_set}"
done

utils/combine_data.sh data/${train_set} ${train_data_dirs}
utils/combine_data.sh data/${dev_set} ${dev_data_dirs}
utils/combine_data.sh data/${eval_set} ${eval_data_dirs}

# remove tmp directories
rm -rf data/p[0-9]*

echo "Successfully prepared data."
