#!/bin/bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

train_set_ori=train_nodup
train_set=train_nodup_sp
train_dev=train_dev
recog_set="eval1 eval2 eval3"

# make segment-wise scp file
for x in "${train_set}" "${train_dev}" ${recog_set}; do
    local/csj_segment_scp.py --segments data/${x}/segments --scp data/${x}/wav.scp > data/${x}/wav_seg.scp
    mv data/${x}/wav.scp data/${x}/wav.scp.bak
    mv data/${x}/wav_seg.scp data/${x}/wav.scp
done
