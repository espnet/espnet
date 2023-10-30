#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="all" # one of all es en fr nl it pt pl de
data_split="10h" # one of full 1h 10h

train_set="mls_${lang}_train"
valid_set="mls_${lang}_dev"
lm_train_text=data/${lang}_lm_train.txt

if [ "$lang" == "all" ]; then
    for ln in es en fr nl it pt pl de; do
        test_sets+="mls_${ln}_test "
    done
else
    test_sets="mls_${lang}_test"
fi

asr_config=conf/tuning/train_asr_e_branchformer1_wavlm_lr1e-4.yaml
inference_config=conf/tuning/decode_transformer_nolm.yaml

ngpu=1

./asr.sh \
    --local_data_opts "--lang ${lang} --data_split ${data_split}" \
    --stage 1 \
    --stop_stage 100 \
    --nj 40 \
    --ngpu ${ngpu} \
    --use_lm true \
    --token_type bpe \
    --nbpe 150 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "${lm_train_text}" "$@"
