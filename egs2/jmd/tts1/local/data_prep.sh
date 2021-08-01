#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi, Takenori Yoshimura
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Prepare kaldi-style data directory for JMD corpus

db=$1
dialect=$2
data_dir=$3
fs=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db> <dialect> <data_dir> <fs>"
    echo "e.g.: $0 downloads kumamoto data/all 24000"
    exit 1
fi

set -euo pipefail

for dset in ${dialect}; do
    # check directory existence
    [ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

    # set filenames
    scp=${data_dir}/wav.scp
    utt2spk=${data_dir}/utt2spk
    spk2utt=${data_dir}/spk2utt
    text=${data_dir}/text
    segments=${data_dir}/segments

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"
    [ -e "${segments}" ] && rm "${segments}"

    # make scp, utt2spk, spk2utt, and segments
    find "${db}/${dset}/wav24kHz" -name "*.wav" | sort | while read -r filename; do
        utt_id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        if [ "${fs}" -eq 24000 ]; then
            # default sampling rate
            echo "${utt_id} ${filename}" >> "${scp}"
        else
            echo "${utt_id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
        fi
        echo "${utt_id} JMD" >> "${utt2spk}"
    done
    utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

    # make text
    local/clean_text.py --skip-header "${db}/${dset}/transcripts.csv" > "${text}"

    # copy segments
    cp "${db}/${dset}/segments" "${segments}"

    # fix
    utils/fix_data_dir.sh "${data_dir}"
    echo "Successfully prepared ${dset}."
done
