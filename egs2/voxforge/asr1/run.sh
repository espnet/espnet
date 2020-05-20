#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=it # de, en, es, fr, it, nl, pt, ru
train_set="tr_${lang}"
dev_set="dt_${lang}"
eval_sets="et_${lang}"

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode_asr.yaml

# FIXME(kamo):
# The results with norm_vars=True is odd.
# I'm not sure this is due to bug.

./asr.sh \
    --local_data_opts "--lang ${lang}" \
    --use_lm false \
    --token_type char \
    --feats_type raw \
    --asr_args "--normalize_conf norm_vars=False " \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text" "$@"
