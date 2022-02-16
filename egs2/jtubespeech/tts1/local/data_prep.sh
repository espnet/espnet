#!/usr/bin/env bash

# Copyright 2021 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

dbr=$1
dbs=$2
data_dir=$3
scorethresh=$4
ROOTDIR="$(cd $(dirname $(dirname $0)); pwd)"

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db_raw> <db_split> <data_dir> <ctcscore_threshold>"
    echo "e.g.: $0 downloads/jtuberaw downloads/jtubesplit data/all -0.5"
    exit 1
fi

set -euo pipefail
. ./path.sh

# split wavfiles
python local/split.py \
    --db_raw ${dbr} \
    --db_split ${dbs}

# prune wavfiles with pre-computed ctcscore
python local/prune.py \
    --score_thresh ${scorethresh} \
    --db_raw ${dbr} \
    --db_split ${dbs}

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

# make text, scp, utt2spk, and spk2utt
cat "${ROOTDIR}/${dbs}/transcript_prune.txt" | sort | while read filename transcript; do
    echo "${filename} ${transcript}" >> "${text}"
    id=${filename}
    spkr=${filename:0:11}
    wavpath=`find ${ROOTDIR}/${dbs} -type f -name ${filename}.wav`
    echo "${id} ${wavpath}" >> "${scp}"
    echo "${id} ${spkr}" >> "${utt2spk}"
done
utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
echo "finished making text, wav.scp, utt2spk, spk2utt."
