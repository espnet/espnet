#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

use_lang_tag=false

. utils/parse_options.sh || exit 1

db=$1
lang=$2
spk=$3
data_dir=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 [options] <download_dir> <lang> <spk> <data_dir>"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
utt2lang=${data_dir}/utt2lang
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}

# make scp, utt2spk, spk2utt, and utt2lang
find ${db} -name "*.wav" -follow | grep ${spk} | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
    echo "${id} ${lang}" >> ${utt2lang}
done
echo "Successfully finished making wav.scp, utt2spk, and utt2lang."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

jsons=$(find ${db}/${lang} -name "*_mls.json" -type f -follow | grep ${spk} | grep -v "/\._" | tr "\n" " ")
${use_lang_tag} && lang_tag=${lang} || lang_tag=""
local/parse_text.py \
    --lang_tag ${lang_tag} \
    --spk_tag ${spk} \
    $(printf "%s" "${jsons[@]}") \
    ${data_dir}/text
echo "Successfully finished making text."
