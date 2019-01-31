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
nj=32

. utils/parse_options.sh || exit 1;

if [ ! $# -eq 2 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir>";
   echo "e.g.: $0 data/train exp/trim_silence/train"
fi

set -euo pipefail
data=$1
logdir=$2

tmpdir=$(mktemp -d ${data}/tmp-XXXX)
split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${tmpdir}/wav.${n}.scp"
done
utils/split_scp.pl ${data}/wav.scp ${split_scps} || exit 1;

# make segments file describing start and end time
${cmd} JOB=1:${nj} ${logdir}.JOB.log \
    local/trim_silence.py \
        --fs ${fs} \
        --win_length ${win_length} \
        --shift_length ${shift_length} \
        --threshold ${threshold} \
        --min_silence ${min_silence} \
        --normalize ${normalize} \
        scp:${tmpdir}/wav.JOB.scp \
        ${tmpdir}/segments.JOB

# concatenate segments
for n in $(seq ${nj}); do
    cat ${tmpdir}/segments.${n} || exit 1;
done > ${data}/segments || exit 1
rm -rf ${tmpdir}

# update utt2spk, spk2utt, and text
[ ! -e ${data}/.backup ] &&  mkdir ${data}/.backup
cp ${data}/{utt2spk,spk2utt,text} ${data}/.backup
paste -d " " \
    <(cut -d " " -f 1 ${data}/segments) \
    <(cut -d " " -f 2 ${data}/.backup/utt2spk) \
    > ${data}/utt2spk
paste -d " " \
    <(cut -d " " -f 1 ${data}/segments) \
    <(cut -d " " -f 2- ${data}/.backup/text) \
    > ${data}/text
utils/utt2spk_to_spk2utt.pl ${data}/utt2spk > ${data}/spk2utt

# check
utils/validate_data_dir.sh --no-feats ${data}
echo "Successfully trimed silence part."
