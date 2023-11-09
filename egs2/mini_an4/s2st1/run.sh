#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


./s2st.sh \
    --nj 2 \
    --inference_nj 2 \
    --src_lang en \
    --tgt_lang es \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --feats_type raw \
    --audio_format flac.ark \
    --inference_config conf/decode_debug.yaml \
    --score_asr_model_tag espnet/kamo-naoyuki_mini_an4_asr_train_raw_bpe_valid.acc.best \
    --train_set "train_nodev" \
    --valid_set "train_dev" \
    --test_sets "train_dev test test_seg" "$@"
