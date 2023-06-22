#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# number of training channels
nch_train=8
# NOTE: The reference signals in spk1.scp in dt_simu_* and et_simu_* subsets
#       are not strictly aligned with signals in wav.scp in the same directory.
#       Be careful when evaluating the performance on these subsets.

train_set="tr_simu_${nch_train}ch_multich"
valid_set="dt_simu_${nch_train}ch_multich"
test_sets="et_simu_${nch_train}ch_multich"
#test_sets="dt_real_${nch_train}ch_multich et_real_${nch_train}ch_multich"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 2 \
    --ref_num 1 \
    --local_data_opts "--nch_train ${nch_train}" \
    --lang en \
    --enh_config conf/tuning/train_enh_beamformer_wpe_mvdr.yaml \
    "$@"
