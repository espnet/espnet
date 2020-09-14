#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set=tr_spatialized_anechoic_multich_${min_or_max}_${sample_rate}
valid_set=cv_spatialized_anechoic_multich_${min_or_max}_${sample_rate}
test_sets="tt_spatialized_anechoic_multich_${min_or_max}_${sample_rate}"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --srctexts "data/${train_set}/text_spk1 data/${train_set}/text_spk2" "$@"
