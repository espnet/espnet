#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k
nj=16
num_spk=2   # one of (2, 3, 4)


train_set=train_si284
valid_set=cv_dev93
test_sets="test_eval92"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --nj ${nj} --num_spk ${num_spk}" \
    --enh_config ./conf/tuning/train_enh_beamformer_no_wpe.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    "$@"
