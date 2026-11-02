#!/usr/bin/env bash

# Data prep for VCTK-Corpus-0.92 (wav48_silence_trimmed/ + txt/, no forced-alignment lab/).
# Adapted from data_prep.sh, which targets the older VCTK-Corpus (0.80) layout.

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=5
num_eval=5
train_set="train"
dev_set="dev"
eval_set="test"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1
dst=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <src-dir> <dst-dir>"
    echo "e.g.: $0 downloads/VCTK-Corpus data"
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

spks=$(find "${db}/wav48_silence_trimmed" -maxdepth 1 -name "p*" -exec basename {} \; | sort)
train_data_dirs=""
dev_data_dirs=""
eval_data_dirs=""
for spk in ${spks}; do
    [ ! -e  ${dst}/${spk}_train ] && mkdir -p ${dst}/${spk}_train

    # set filenames
    scp=${dst}/${spk}_train/wav.scp
    utt2spk=${dst}/${spk}_train/utt2spk
    text=${dst}/${spk}_train/text
    spk2utt=${dst}/${spk}_train/spk2utt

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"

    # make scp, text (mic1 only: p280 has no mic2 recordings, so use mic1 for every speaker)
    find "${db}/wav48_silence_trimmed/${spk}" -follow -name "*_mic1.flac" | sort | while read -r wav; do
        id=$(basename "${wav}" _mic1.flac)
        txt=${db}/txt/${spk}/${id}.txt

        if [ ! -e "${txt}" ]; then
            echo "${id} does not have a text file. skipped."
            continue
        fi

        echo "${id} ${wav}" >> "${scp}"
        echo "${id} ${spk}" >> "${utt2spk}"
        echo "${id} $(cat "${txt}" | tr -d '\r')" >> "${text}"
    done

    if [ ! -s "${scp}" ]; then
        echo "${spk} has no utterance with a text file. Skipping speaker."
        rm -rf "${dst}/${spk}_train"
        continue
    fi
    utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

    # split
    num_all=$(wc -l < "${scp}")
    if [ -z "${dev_set}" ]; then
        # no devset, use all data
        train_data_dirs+=" ${dst}/${spk}_train"
    else
        # need to divide into train/dev/eval
        num_deveval=$((num_dev + num_eval))
        num_train=$((num_all - num_deveval))

        utils/subset_data_dir.sh --last "${dst}/${spk}_train" "${num_deveval}" "${dst}/${spk}_deveval"
        utils/subset_data_dir.sh --first "${dst}/${spk}_deveval" "${num_dev}" "${dst}/${spk}_${dev_set}"
        utils/subset_data_dir.sh --last "${dst}/${spk}_deveval" "${num_eval}" "${dst}/${spk}_${eval_set}"
        utils/subset_data_dir.sh --first "${dst}/${spk}_train" "${num_train}" "${dst}/${spk}_${train_set}"

        # remove tmp directories
        rm -rf "${dst}/${spk}_train"
        rm -rf "${dst}/${spk}_deveval"
        train_data_dirs+=" ${dst}/${spk}_${train_set}"
        dev_data_dirs+=" ${dst}/${spk}_${dev_set}"
        eval_data_dirs+=" ${dst}/${spk}_${eval_set}"

    fi
done

utils/combine_data.sh ${dst}/${train_set} ${train_data_dirs}
utils/fix_data_dir.sh ${dst}/${train_set}

if [ ! -z "${dev_set}" ]; then
    utils/combine_data.sh ${dst}/${dev_set} ${dev_data_dirs}
    utils/combine_data.sh ${dst}/${eval_set} ${eval_data_dirs}

    utils/fix_data_dir.sh ${dst}/${dev_set}
    utils/fix_data_dir.sh ${dst}/${eval_set}
fi

# remove tmp directories
rm -rf ${dst}/p[0-9]*

echo "Successfully prepared data."
