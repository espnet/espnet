#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=6
train_set=
valid_set=
test_sets=
enh_config=
extra_annotations=
ref_channel=

. ../utils/parse_options.sh

dir=$PWD
cd ../../enh1

./enh.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs 16k \
    --ngpu 1 \
    --ref_num 1 \
    --ref_channel "${ref_channel}" \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 1 --stop-stage 2" \
    --enh_config "${enh_config}" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth"

cd $dir
