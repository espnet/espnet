#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=max # Must be "max" for asr. This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set="tr_${min_or_max}_${sample_rate}"
valid_set="cv_${min_or_max}_${sample_rate}"
test_sets="tt_${min_or_max}_${sample_rate}"

./asr_mix.sh \
    --lang "en" \
    --nbpe 5000 \
    --max_wav_duration 15 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/tuning/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ngpu 4 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"
    "$@"