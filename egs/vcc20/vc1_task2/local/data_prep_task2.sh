#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2
lang=$3
spk=$4
trans_type=$5

# check arguments
if [ $# != 5 ]; then
    echo "Usage: $0 <db> <data_dir> <lang> <spk> <trans_type>"
    exit 1
fi

# check speaker
available_spks=(
    "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
)

if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified speaker ${spk} is not available."
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
lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
find ${db}/${spk} -follow -name "${lang_char}[12]*.wav" | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text (only for the utts in utt2spk)
case "${lang_char}" in
    "M")
        lang_tag=zh_ZH
        local/clean_text_mandarin.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} ${spk} > ${text}
        ;;
    "F")
        lang_tag=fi_FI
        local/clean_text_finnish.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} ${spk} > ${text}
        ;;
    "G")
        lang_tag=de_DE
        local/clean_text_german.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} ${spk}> ${text}
        ;;
esac
echo "finished making text."
