#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=it # de, en, es, fr, it, nl, pt, ru
train_set="tr_${lang}"
valid_set="dt_${lang}"
test_sets="dt_${lang} et_${lang}"

asr_config=conf/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr.yaml

# FIXME(kamo):
# The results with norm_vars=True is odd.
# I'm not sure this is due to bug.

./asr.sh \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm false \
    --token_type char \
    --feats_type raw \
    --asr_args "--normalize_conf norm_vars=False " \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
