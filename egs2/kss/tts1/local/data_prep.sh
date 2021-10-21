#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db_root> <data_dir>"
    echo "e.g.: $0 /path/to/kss data/train"
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
find "${db_root}" -name "*.wav" | sort | while read -r filename; do
    id=kss_$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    # NOTE(kan-bayashi): Some wav files are stereo
    echo "${id} sox ${filename} -t wav -c 1 - |" >> "${scp}"
    echo "${id} kss" >> "${utt2spk}"
done
utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
echo "Successfully finished making wav.scp, utt2spk, spk2utt."

# make text
find "${db_root}" -name "transcript*.txt" | sort | while read -r filename; do
   sed < "${filename}" -e "s;^[1-4]/;;g" -e "s/.wav//g" |
        awk -F "|" -v spk=kss '{print spk "_" $1 " " $2}' |
        sort >> "${text}"
done
echo "Successfully finished making text."

utils/fix_data_dir.sh "${data_dir}"
echo "Successfully finished preparing data directory."
