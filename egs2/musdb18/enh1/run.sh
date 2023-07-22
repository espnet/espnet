#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=44.1k
use_hq=false    # true to download wav format; false to download mp4 format
nchannels=1     # number of channels of the data (1 or 2)


train_set="train_${sample_rate}"
valid_set="dev_${sample_rate}"
test_sets="test_${sample_rate} "

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --audio_format wav \
    --max_wav_duration 1000 \
    --ref_num 4 \
    --lang en \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --use_hq ${use_hq} --nchannels ${nchannels}" \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    "$@"
