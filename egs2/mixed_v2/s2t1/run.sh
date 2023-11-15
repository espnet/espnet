#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="test/LibriSpeech/test_clean test/LibriSpeech/test_other test/SWBD/eval2000 test/TEDLIUM/test"

nbpe=50000
s2t_config=conf/tuning/train_s2t_transformer_conv2d_size1024_e18_d18_lr5e-4_warmup20k.yaml
inference_config=conf/decode_s2t.yaml

# inference only args
# --cleaner whisper_en --hyp_cleaner whisper_en
./s2t.sh \
    --stage 1 \
    --stop_stage 13 \
    --use_lm false \
    --num_nodes 32 \
    --ngpu 4 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 64 \
    --num_splits_s2t 10 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 10000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
