#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2
data_dir=$3
trans_type=$4
lang=$5
dataset=$6

# check arguments
if [ $# != 6 ]; then
    echo "Usage: $0 <db> <spk> <data_dir> <trans_type> <lang_tag> <dataset (train/dev)>"
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

lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
trg_lang_char=$(echo ${lang} | head -c 1)

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text
feat=${data_dir}/feats.scp

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${feat} ] && rm ${feat}

# 20200202
# We need utt2spk for making text
# We need feat.scp
# to make input 1
# so that we can put x-vector as input 2
# so that IO can work normally...
list_path=${db}/lists/${trg_lang_char}_${dataset}_list.txt
dummy_ark=/home/huang18/VC/Experiments/espnet/egs/vcc2020/tts1/fbank/raw_fbank_TMF1.1.ark:7
sed "s/$/ $spk/" ${list_path} > ${utt2spk}
sed "s~$~ $dummy_ark~" ${list_path} > ${feat}
echo "finished making feats.scp, utt2spk"

# make text (only for the utts in utt2spk)
# language dependent
case "${trg_lang_char}" in
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
            --lang_tag ${lang_tag} \
            ${db}/prompts/${lang}_transcriptions.txt \
            ${utt2spk} $trans_type > ${text}
        ;;
    *)
        echo "We don't have a text cleaner for this language now.";
        continue
        ;;
esac
echo "finished making text."
