#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k



train_set=tr_synthetic
valid_set=cv_synthetic
test_sets="tt_synthetic_no_reverb tt_synthetic_with_reverb"

./enh.sh \
    --lang en \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --ref_num 1 \
    --local_data_opts "" \
    --enh_config ./conf/tuning/train_enh_blstm_tf.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --max_wav_duration 31 \
    --inference_model "valid.loss.best.pth" \
    "$@"
