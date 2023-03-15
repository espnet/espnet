#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set="train"
dev_set="dev"

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

db_root=$1
list_root=$2
data_dir=$3
task=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db_root> <list_root> <data_dir> <task1/task2>"
    exit 1
fi

if [ ${task} = "task1" ]; then
    available_spks=(
        "SEF1" "SEF2" "SEM1" "SEM2" "TEF1" "TEF2" "TEM1" "TEM2"
    )
    available_langs=(
        "Eng"
    )
elif [ ${task} = "task2" ]; then
    available_spks=(
        "SEF1" "SEF2" "SEM1" "SEM2" "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
    )
    available_langs=(
        "Eng" "Ger" "Fin" "Man"
    )
else
    echo "Task can only be either task1 or task2."
fi

# make dirs
for name in all "${train_set}" "${dev_set}" "${eval_set}"; do
    [ ! -e "${data_dir}/${name}" ] && mkdir -p "${data_dir}/${name}"
done

# set filenames
scp="${data_dir}/all/wav.scp"
train_scp="${data_dir}/${train_set}/wav.scp"
dev_scp="${data_dir}/${dev_set}/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${train_scp}" ] && rm "${train_scp}"
[ -e "${dev_scp}" ] && rm "${dev_scp}"

for spk in "${available_spks[@]}"
do
    # make scp
    find "${db_root}/${spk}" -name "*.wav" -follow | sort | while read -r filename;do
        id="${spk}_$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
        echo "${id} ${filename}" >> "${scp}"
    done
done

# make train/dev set
for lang in "${available_langs[@]}"
do
    lang_char=$(echo ${lang} | head -c 1)
    local/subset_data_dir.py --utt_list ${list_root}/${lang_char}_train_list.txt --scp ${scp} >> ${train_scp}
    local/subset_data_dir.py --utt_list ${list_root}/${lang_char}_dev_list.txt --scp ${scp} >> ${dev_scp}
done

# remove all
rm -rf "${data_dir}/all"

echo "successfully prepared data."
