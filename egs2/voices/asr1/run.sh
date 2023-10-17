#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

use_devkit_subsets=false  # true to use the full dataset, false to use devkit subsets
use_multich=false  # true to use the multi-channel versionl, false to use the single-channel data
if $use_devkit_subsets; then
    if $use_multich; then
        suffix="_2ch"
    else
        suffix=""
    fi
    train_set=devkit_${train_set}${suffix}
    valid_set=devkit_${valid_set}${suffix}
    test_set=devkit_${test_set}${suffix}
else
    if $use_multich; then
        suffix="_multich"
    else
        suffix=""
    fi
    train_set=full_${train_set}${suffix}
    valid_set=full_${valid_set}${suffix}
    test_set=full_${test_set}${suffix}
fi

asr_config=conf/train_asr_conformer.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --local_data_opts "--use_devkit_subsets ${use_devkit_subsets}" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
