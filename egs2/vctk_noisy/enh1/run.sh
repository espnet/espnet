#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k



train_set=tr_26spk
valid_set=cv_2spk
test_sets=tt_2spk

./enh.sh \
    --lang en \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 1 \
    --local_data_opts "" \
    --enh_config ./conf/train.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --max_wav_duration 30 \
    --inference_model "valid.loss.best.pth" \
    "$@"
