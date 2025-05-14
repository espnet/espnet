#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

test_sets="test_l1 test_l2"

nbpe=50000
s2t_exp=espnet/owsm_v3.1_ebf
inference_config=conf/decode_s2t.yaml

./s2t.sh \
    --use_lm false \
    --gpu_inference true \
    --token_type bpe \
    --nbpe ${nbpe} \
    --inference_config "${inference_config}" \
    --skip_train true \
    --test_sets "${test_sets}" \
    --download_model $s2t_exp \
    --cleaner whisper_en \
    --hyp_cleaner whisper_en \
    --stop_stage 13 \
    --skip_stages "2 4 5 6 7 8 9 10 11" \
