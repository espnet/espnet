#!/bin/bash
# Append the dataset name to the speaker name in utt2spk and spk2utt files

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <data-dir> <dataset_name>"
    echo "e.g.: $0 ../../data/tr_no_dev kising"
    exit 1
fi

dir=$1
dataset_name=$2


utt2spk="${dir}/utt2spk"
utt2spk_temp="${dir}/utt2spk.temp"
if [ -f "$utt2spk_temp" ]; then
    rm "$utt2spk_temp"
fi

echo "Fixing spk in $utt2spk"
while read -r utt spk; do
    if [[ $utt == "${dataset_name}_${spk}"* && $spk != "${dataset_name}"* ]]; then
        spk="${dataset_name}_${spk}"
    fi
    echo "$utt $spk" >> "$utt2spk_temp"
done < "$utt2spk"

mv "$utt2spk_temp" "$utt2spk"

utils/utt2spk_to_spk2utt.pl < ${dir}/utt2spk > ${dir}/spk2utt
