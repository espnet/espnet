#!/usr/bin/env bash

# Copyright 2021 Peter Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set=train_no_dev
dev_set=dev
eval_set=eval

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1
spk=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db> <spk>"
    echo ""
    echo "Options:"
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    exit 1
fi

# check speaker
available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
)
if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified speaker ${spk} is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

set -euo pipefail

[ ! -e data/${spk} ] && mkdir -p data/${spk}

# set filenames
scp=data/${spk}/wav.scp
utt2spk=data/${spk}/utt2spk
text=data/${spk}/text
segments=data/${spk}/segments
spk2utt=data/${spk}/spk2utt

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"
[ -e "${segments}" ] && rm "${segments}"

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
raw_text=${db}/etc/txt.done.data
ids=$(sed < ${raw_text} -e "s/^( /${spk}_/g" -e "s/ )$//g" | cut -d " " -f 1)
sentences=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" | tr '[:lower:]' '[:upper:]' | cut -d " " -f 2-)
paste -d " " <(echo "${ids}") <(echo "${sentences}") > ${text}
echo "Successfully finished making text."

utils/fix_data_dir.sh data/${spk}
utils/validate_data_dir.sh --no-feats data/${spk}

# split
utils/subset_data_dir.sh --last data/${spk} 200 data/${spk}_tmp
utils/subset_data_dir.sh --last data/${spk}_tmp 100 data/${eval_set}
utils/subset_data_dir.sh --first data/${spk}_tmp 100 data/${dev_set}
n=$(( $(wc -l < data/${spk}/wav.scp) - 200 ))
utils/subset_data_dir.sh --first data/${spk} ${n} data/${train_set}

# remove tmp directories
rm -rf data/${spk}_tmp

echo "Successfully prepared data."
