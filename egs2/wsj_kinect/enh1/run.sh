#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

### The following two parameters are currently fixed for Kinect-WSJ
#min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=16k
parallel=true
use_dereverb=false


train_set="tr"
valid_set="cv"
test_sets="tt"

./enh.sh \
    --ref_num 2 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang en \
    --ngpu 2 \
    --local_data_opts "--parallel ${parallel} --use_dereverb ${use_dereverb}" \
    --enh_config conf/train.yaml \
    "$@"
