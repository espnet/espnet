#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Takenori Yoshimura)
#           2021 Nagoya University (Yusuke Yasuda)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
fs=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <fs>"
    echo "e.g.: $0 downloads/J-KAC data/all 24000"
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
wav=${data_dir}/wav
segments=${data_dir}/segments

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"

local/prep_segments.py ${db} ${scp} ${utt2spk} ${text} ${segments} $fs
echo "finished making wav.scp, utt2spk, text, segments."

utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
echo "finished making spk2utt."
