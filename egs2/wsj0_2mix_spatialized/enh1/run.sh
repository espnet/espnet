#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set=tr_spatialized_anechoic_multich
dev_set=cv_spatialized_anechoic_multich
eval_sets="tt_spatialized_anechoic_multich "

./enh.sh \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --fs 8k \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_beamformer_tf1.0.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    "$@"
