#!/usr/bin/env bash

# Copyright 2022  Carnegie Mellon University (Authors: Xuankai Chang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)


min_or_max=min
sample_rate=8k

local_data_args="$*"


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

local/data_supervised.sh ${local_data_args}

sup_train_set="tr_"${min_or_max}_${sample_rate}
train_set="tr_"${min_or_max}_${sample_rate}_w_1spk_utt

# Create a dataset for pseudo semi-supervised learning
# We only use them for the 1-speaker speech, but still in MixIT loss
# How to use them for supervised learning (PIT loss) is to be developed.
cp -r data/${sup_train_set}/ data/${train_set}

random_chosen_utt="data/${train_set}/random_chosen_utt_list"
shuf -n 8000 data/${sup_train_set}/wav.scp \
    > data/${train_set}/random_chosen_utt_list || exit 1;

# Here we just include 8000 utterance of 1-speaker training data
# Note that this part can be changed to other 1-speaker speech data as well
for f in wav.scp spk1.scp spk2.scp; do
    <${random_chosen_utt} sed 's/data\/wav/data\/spk1/' | \
        awk '{print($1"_single", $2)}' >> data/${train_set}/${f} || exit 1;
done

awk '(ARGIND==1){utt2spk[$1]=$2}
     (ARGIND==2){print($1"_single", utt2spk[$1])}' \
    data/${sup_train_set}/utt2spk ${random_chosen_utt} >> data/${train_set}/utt2spk

# utt2category
<data/${sup_train_set}/utt2spk awk '{print($1, "2speaker")}' > data/${train_set}/utt2category
<${random_chosen_utt} \
    awk '{print($1"_single", "1speaker")}' \
    >> data/${train_set}/utt2category

utils/fix_data_dir.sh --utt_extra_files "spk1.scp spk2.scp utt2category" data/${train_set}
for f in spk1.scp spk2.scp utt2category; do
    check_sorted data/${train_set}/${f}
done

utils/utt2spk_to_spk2utt.pl data/${train_set}/utt2spk > data/${train_set}/spk2utt

rm ${random_chosen_utt}
