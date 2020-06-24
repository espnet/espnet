#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_spatialized_reverb_multich
dev_set=cv_spatialized_reverb_multich
eval_sets="tt_spatialized_reverb_multich "

./enh.sh \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --ngpu 1 \
    --enh_config ./conf/tuning/train_enh_beamformer_tf1.0.yaml \
    --use_dereverb_ref true \
    --use_noise_ref false \
    "$@"
