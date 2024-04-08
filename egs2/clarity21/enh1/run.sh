#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16000 # by default we resample to 16k

# put the path here to the clarity first enhancement challenge folder which contains
# dev  hrir  metadata  train subfolders
clarity_root=/raid/users/popcornell/Clarity/target_dir/clarity_CEC1_data/clarity_data/

train_set=train
valid_set=dev
test_sets="dev"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 1 \
    --ref_channel 0 \
    --local_data_opts "--clarity_root ${clarity_root} --sample_rate ${sample_rate}" \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    "$@"
