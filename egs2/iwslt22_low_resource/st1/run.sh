#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=ta
tgt_lang=fr

# train: 17 hours of Tamasheq audio data aligned to French translations
# train_full: a 19 hour version of this corpus, 
# including 2 additional hours of data that was labeled by annotators as potentially noisy 
train_set=train
train_dev=valid
test_set=test

st_config=conf/train_st_transformer.yaml
inference_config=conf/decode_st_transformer.yaml

tgt_nbpe=1000

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
# Note, it is best to keep tgt_case as tc to match IWSLT22 eval
tgt_case=tc

./st.sh \
		--st_tag wav2vec-transformer-warmup-15k \
    --ignore_init_mismatch true \
    --stage 1 \
    --stop_stage 13 \
    --use_lm false \
    --token_joint false \
    --audio_format "wav" \
    --nj 16 \
    --inference_nj 16 \
    --src_lang ${src_lang} \
		--use_src_lang false \
    --tgt_lang ${tgt_lang} \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --tgt_case ${tgt_case} \
    --feats_type "raw" \
		--feats_normalize uttmvn \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@"
