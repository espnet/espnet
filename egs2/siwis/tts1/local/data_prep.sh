#!/usr/bin/env bash

# Copyright 2021 Takenori Yoshimura
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1
data_dir=$2
subsets=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <data_dir> <subset>"
    echo "e.g.: $0 downloads/SIWIS data/train 1,2,3"
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
segments=${data_dir}/segments

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"
[ -e "${segments}" ] && rm "${segments}"

# make files
for part in ${subsets//,/ }; do
    while read -r filename; do
        id=$(basename "${filename}" .lab)
        echo "${id} SIWIS" >> "${utt2spk}"

        wavname="${db_root}/wavs/part${part}/${id}.wav"
        echo "${id} ${wavname}" >> "${scp}"

        txtname="${db_root}/text/part${part}/${id}.txt"
        {
            echo -n "${id} "
            cat "${txtname}"
            echo ""
        } >> "${text}"

        labname="${db_root}/labs/part${part}/${id}.lab"
        s=$(head -n 1 "${labname}" | cut -d" " -f 2 | awk '{ print $1 / 10000000 }')
        e=$(tail -n 1 "${labname}" | cut -d" " -f 1 | awk '{ print $1 / 10000000 }')
        echo "${id} ${id} ${s} ${e}" >> "${segments}"
    done < "${db_root}/lists/lab.part${part}_all.list"
done

sort "${utt2spk}" -o "${utt2spk}"
utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"

# replace UTF-8 white spaces and remove empty lines
# https://stackoverflow.com/questions/43638993/bash-remove-all-unicode-spaces-and-replace-with-normal-space
cp "${text}" "${text}.bak"
perl -CSDA -plE 's/\s/ /g' "${text}.bak" | sed '/^$/d' > "${text}"
rm "${text}.bak"

utils/fix_data_dir.sh "${data_dir}"
echo "Successfully finished preparing data directory."
