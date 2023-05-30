#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev"

nbpe=50000
s2t_config=conf/tuning/train_s2t_transformer_conv2d_size1024_e24_d24_lr1e-3_warmup30k.yaml
inference_config=conf/decode_s2t.yaml

./s2t.sh \
    --stage 5 \
    --stop_stage 5 \
    --use_lm false \
    --num_nodes 8 \
    --ngpu 4 \
    --nj 128 \
    --gpu_inference true \
    --inference_nj 32 \
    --num_splits_s2t 5 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
