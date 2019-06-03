#!/bin/bash -e

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

# check language
if [ ${lang} = "de_DE" ]; then
    available_spks=("angela" "rebecca" "ramona" "eva" "karlsson")
elif [ ${lang} = "en_UK" ]; then
    available_spks=("elizabeth")
elif [ ${lang} = "it_IT" ]; then
    available_spks=("lisa" "riccardo")
elif [ ${lang} = "es_ES" ]; then
    available_spks=("karen" "tux" "victor")
elif [ ${lang} = "en_US" ]; then
    available_spks=("judy" "mary" "elliot")
elif [ ${lang} = "fr_FR" ]; then
    available_spks=("ezwa" "nadine" "bernard" "gilles" "zeckou")
elif [ ${lang} = "uk_UK" ]; then
    available_spks=("sumska" "loboda" "miskun" "obruchov" "shepel")
elif [ ${lang} = "ru_RU" ]; then
    available_spks=("hajdurova" "minaev" "nikolaev")
else
    echo "${lang} is not supported."
    exit 1
fi

# check speaker
if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified speaker is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" -follow | grep ${spk} | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

jsons=$(find ${db}/${lang} -name "*_mls.json" -type f -follow | grep ${spk} | grep -v "/._" | tr "\n" " ")
${use_lang_tag} && lang_tag=${lang} || lang_tag=""
local/parse_text.py \
    --lang_tag ${lang_tag} \
    --spk_tag ${spk} \
    $(printf "%s" "${jsons[@]}") \
    ${data_dir}/text
echo "Successfully finished making text."
