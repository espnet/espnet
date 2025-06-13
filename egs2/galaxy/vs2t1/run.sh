#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=galaxy_train
valid_set=galaxy_val
test_sets="galaxy_test"

nbpe=50000
s2t_config=conf/train_setting7_clip_large_bz8_ag128_ngpu1.yaml
# s2t_config=conf/state_data.yaml

inference_config=conf/decode_s2t.yaml
s2t_stats_dir=exp/s2t_stats_raw_bpe50000_clip_setting7

./vs2t.sh \
    --s2t_stats_dir "${s2t_stats_dir}" \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 32 \
    --num_splits_s2t 12 \
    --dumpdir "data" \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --inference_s2t_model valid.total_count.ave_5best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "data/raw/${train_set}/text" "$@"
