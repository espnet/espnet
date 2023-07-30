#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./s2t.sh \
    --use_lm false \
    --num_nodes 1 \
    --nj 2 \
    --inference_nj 2 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe 1537 \
    --bpe_input_sentence_size 10000000 \
    --s2t_config "conf/train_transformer.yaml" \
    --inference_config "conf/decode_s2t.yaml" \
    --train_set "train_nodev" \
    --valid_set "train_dev" \
    --test_sets "test" \
    --bpe_train_text "dump/raw/train_nodev/text" \
    --bpe_nlsyms "data/nlsyms.txt" \
    --lm_train_text "dump/raw/train_nodev/text" "$@"
