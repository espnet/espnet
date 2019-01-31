#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

fs=16000
win_length=1024
shift_length=256
threshold=60
min_silence=0.01
normalize=16
cmd=run.pl

. parse_options.sh || exit 1;

if [ ! $# -eq 2 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir>";
   echo "e.g.: $0 data/train exp/trim_silence/train"
fi

set -euo pipefail
data=$1
logdir=$2

# make segments file describing start and end time
${cmd} ${logdir}.log \
    local/trim_silence.py \
        --fs ${fs} \
        --win_length ${win_length} \
        --shift_length ${shift_length} \
        --threshold ${threshold} \
        --min_silence ${min_silence} \
        --normalize ${normalize} \
        scp:${data}/wav.scp \
        ${data}/segments

# update utt2spk, spk2utt, and text
[ ! -e ${data}/.backup ] &&  mkdir ${data}/.backup
cp ${data}/{utt2spk,spk2utt,text} ${data}/.backup
awk -v s=${spk} '{printf "%s %s\n",$1,s}' ${data}/segments \
    > ${data}/utt2spk
utils/utt2spk_to_spk2utt.pl ${data}/utt2spk > ${data}/spk2utt
paste -d " " \
    <(cut -d " " -f 1 ${data}/segments) \
    <(cut -d " " -f 2- ${data}/.backup/text) \
    > ${data}/text

# check
utils/validate_data_dir.sh ${data}
echo "Successfully trimed silence part."
