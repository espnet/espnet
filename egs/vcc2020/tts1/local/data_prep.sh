#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2
data_dir=$3
trans_type=$4
lang=$5

# check arguments
if [ $# != 5 ]; then
    echo "Usage: $0 <db> <spk> <data_dir> <trans_type> <lang_tag>"
    exit 1
fi

# check speaker
available_spks=(
    "SEF1" "SEF2" "SEM1" "SEM2" "TEF1" "TEF2" "TEM1" "TEM2" "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
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
find ${db}/${spk} -follow -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text (only for the utts in utt2spk)
# language dependent
lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
case "${lang_char}" in
    "M") 
        lang_tag=zh_ZH
        local/clean_text_mandarin.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${db}/prompts/Eng_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} > ${text}
        ;;
    "E")
        lang_tag=en_US
        local/clean_text_english.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} > ${text}
        ;;
    "F") 
        lang_tag=fi_FI
        local/clean_text_finnish.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${db}/prompts/Eng_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} > ${text}
        ;;
    "G") 
        lang_tag=de_DE
        local/clean_text_german.py \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${db}/prompts/Eng_transcriptions.txt \
            ${utt2spk} $trans_type ${lang_tag} > ${text}
        ;;
    *)
        echo "We don't have a text cleaner for this language now.";
        ;;
esac
echo "finished making text."
