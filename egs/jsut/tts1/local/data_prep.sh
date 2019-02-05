#!/bin/bash

# Copyright 2018 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
data_dir=$2

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
rawtext=${data_dir}/rawtext
text=${data_dir}/text
usr_dict_dir=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
esp_dict_dir=../../../tools/mecab/mecab-ipadic-neologd

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${rawtext} ] && rm ${rawtext}
if [ ! -e ${usr_dict_dir} -a ! -e ${esp_dict_dir} ]; then
    echo "$0: Error: The mecab-ipadic-NEologd can not be found" >&2
    echo "$0: Error: Please modify the dictionary path in this script or" >&2
    echo "$0: Error: install it by running 'make mecab.done' at espnet/tools" >&2
    exit 1
fi

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${filename}" >> ${scp}
    echo "${id} JS" >> ${utt2spk}
done
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# make text
find ${db} -name "transcript_utf8.txt" | sort | while read -r filename; do
    cat ${filename} >> ${rawtext}
done
if [ -e ${usr_dict_dir} ]; then
   dict=${usr_dict_dir}
else
   dict=${esp_dict_dir}
fi
PYTHONIOENCODING=utf-8 PYTHONPATH=local/text python local/clean_text.py \
   ${dict} ${rawtext} > ${text}
rm ${rawtext}
echo "finished making text."
