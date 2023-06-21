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

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 [Options] <db>"
    echo ""
    echo "Options:"
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    exit 1
fi

set -euo pipefail

[ ! -e data/all ] && mkdir -p data/all

# set filenames
scp=data/all/wav.scp
utt2spk=data/all/utt2spk
text=data/all/text
segments=data/all/segments
spk2utt=data/all/spk2utt

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"
[ -e "${segments}" ] && rm "${segments}"

# make scp, utt2spk, and spk2utt
for spk in slt clb bdl rms jmk awb ksp
do
    find ${db} -name "$spk*.wav" -follow | sort | while read -r filename;do
        id="$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
        echo "${id} ${filename}" >> ${scp}
        echo "${id} ${spk}" >> ${utt2spk}
    done
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
raw_text=${db}/etc/txt.done.data
ids=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" | cut -d " " -f 1)
sentences=$(sed < ${raw_text} -e "s/^( //g" -e "s/ )$//g" -e "s/\"//g" | tr '[:lower:]' '[:upper:]' | cut -d " " -f 2-)
paste -d " " <(echo "${ids}") <(echo "${sentences}") > ${text}
echo "Successfully finished making text."

utils/fix_data_dir.sh data/all
utils/validate_data_dir.sh --no-feats data/all

#create utt-list
python local/get_utt_list.py


# split
utils/subset_data_dir.sh --utt-list "data/all/utt_train_list.txt" data/all data/${train_set}
utils/subset_data_dir.sh --utt-list "data/all/utt_dev_list.txt" data/all data/${dev_set}
utils/subset_data_dir.sh --utt-list "data/all/utt_eval_list.txt" data/all data/${eval_set}


echo "Successfully prepared data."