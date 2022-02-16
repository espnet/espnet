#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
fs=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <fs>"
    echo "e.g.: $0 downloads/jsut_ver1.1 data/all 24000"
    exit 1
fi

set -euo pipefail

# check directory existence
[ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"

# make scp, utt2spk, and spk2utt
find "${db}" -name "*.wav" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    if [ "${fs}" -eq 48000 ]; then
        # default sampling rate
        echo "${id} ${filename}" >> "${scp}"
    else
        echo "${id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
    fi
    echo "${id} JS" >> "${utt2spk}"
done
utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
echo "finished making wav.scp, utt2spk, spk2utt."

# make text
find ${db} -name "transcript_utf8.txt" | sort | while read -r filename; do
    tr ':' ' ' < "${filename}" | sort >> "${text}"
done

echo "finished making text."
