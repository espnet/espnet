#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=dev

nbpe=50000
s2t_config=conf/train_conv2d8_size1024_e18_d18_mel128.yaml

./s2t.sh \
    --use_lm false \
    --num_nodes 4 \
    --ngpu 4 \
    --nj 512 \
    --gpu_inference true \
    --inference_nj 64 \
    --num_splits_s2t 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 100000000 \
    --s2t_config "${s2t_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
