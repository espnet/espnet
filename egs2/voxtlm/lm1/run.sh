#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

nbpe=10000
lm_config=conf/train_transformer_size768_e12.yaml
lm_inference_asr_config=conf/decode_lm_asr.yaml
lm_inference_tts_config=conf/decode_lm_tts.yaml

./lm.sh \
    --stage 7 \
    --stop_stage 9 \
    --num_splits_lm 30 \
    --nj 8 \
    --ngpu 4 \
    --gpu_inference true \
    --inference_nj 8 \
    --lang en \
    --token_type bpe \
    --nbpe "${nbpe}" \
    --bpe_nlsyms data/nlsyms.txt \
    --bpe_train_text "data/${train_set}/bpe_text" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_lm valid.acc.ave.pth \
    --km_dir "/swl/home/smaiti/speechlmscore_tool/models/hubert/" \
    --lm_inference_asr_config "${lm_inference_asr_config}" \
    --lm_inference_tts_config "${lm_inference_tts_config}" \
    --lm_test_text_asr dump/raw/${test_sets}/text.asr \
    --lm_test_text_tts dump/raw/${test_sets}/text.tts \
    --lm_test_text_textlm dump/raw/${test_sets}/text.textlm \
    --lm_test_text_unitlm dump/raw/${test_sets}/text.unitlm "$@"
