#!/usr/bin/env bash

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db-dir> <src-dir> <dst-dir>"
    echo "e.g.: $0 /mls/jubang/databases/KsponSpeech data/local/KsponSpeech data/train"
    exit 1
fi

db=$1
src=$2
dst=$3

data=$(echo $dst | sed 's:\.:/:' | awk -v src=$src -F"/" '{print src"/"$NF"/text.trn"}')
temp=tmp

mkdir -p ${dst} ${dst}/$temp || exit 1;

[ ! -d ${db} ] && echo "$0: no such directory ${db}" && exit 1;
[ ! -f ${data} ] && echo "$0: no such file ${data}. please re-run the script of 'local/trans_prep.sh'." && exit 1;

wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
text=${dst}/text; [[ -f "${text}" ]] && rm ${text}
utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}

# 1) extract meta data
cat $data | cut -f1 -d' ' > ${dst}/${temp}/pcm.list
cat ${dst}/${temp}/pcm.list | awk -F"/" '{print $NF}' | awk -F"." '{print $1}' > ${dst}/${temp}/labels
awk -v db=$db '{print db "/" $0}' ${dst}/${temp}/pcm.list | \
    paste -d' ' ${dst}/${temp}/labels - | sort > ${dst}/${temp}/pcm.scp

# 2) prepare wav.scp
awk '{print $1 " sox -r 16000 -b 16 -c 1 -e signed-integer -t raw " $2 " -t wav - |"}' \
    ${dst}/${temp}/pcm.scp > ${dst}/wav.scp

# 3) prepare text
cat $data | cut -d' ' -f3- > ${dst}/${temp}/text.org
cat ${dst}/${temp}/text.org | local/lowercase.perl | local/remove_punctuation.pl | paste -d' ' ${dst}/${temp}/labels - | sort > ${dst}/text

# 4) prepare utt2spk & spk2utt
spk2utt=${dst}/spk2utt
awk '{print $1 " " $1}' ${dst}/${temp}/labels | sort -k 1 > $utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl < ${utt2spk} > $spk2utt || exit 1

ntext=$(wc -l <$text)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntext" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntext) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in ${dst}"
exit 0;
