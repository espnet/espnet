#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=en
tgt_lang=hi  # one of hi (Hindi), bn (Bengali), or ta (Tamil)

train_set=train.en-${tgt_lang}
train_dev=dev.en-${tgt_lang}
test_set=tst-COMMON.en-${tgt_lang}

st_config=conf/tuning/train_st_conformer.yaml
inference_config=conf/tuning/decode_st_conformer.yaml

./st.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --local_data_opts "${tgt_lang}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --feats_type raw \
    --audio_format "flac.ark" \
    --src_token_type "bpe" \
    --src_nbpe 4000 \
    --tgt_token_type "bpe" \
    --tgt_nbpe 4000 \
    --feats_normalize "utterance_mvn" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --gpu_inference true \
    "$@"
