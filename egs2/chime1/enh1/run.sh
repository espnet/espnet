#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k  # 16k or 48k
# NOTE1: The reference signals in spk1.scp in devel and test subsets are reverberated.
# NOTE2: Dynamic mixing should be used in training data, as the one-to-one noise list
#        is not provided for the training data.

train_set=train_${sample_rate}
valid_set=devel_${sample_rate}
test_sets="test_${sample_rate}"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 1 \
    --local_data_opts "--sample_rate ${sample_rate}" \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
