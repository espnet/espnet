# !/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

timestamp=$(date "+%Y%m%d.%H%M%S")
    # --dumpdir "dump.${timestamp}" \
    # --expdir "exp.${timestamp}" \

./asr.sh \
    --feats_normalize uttmvn \
    --stage 11 \
    --stop_stage 11 \
    --asr_config conf/dcase.yaml \
    --local_score_opts exp/asr_branchformer_raw_en_word/inference_beam_size10_ctc_weight0.3_asr_model_valid.acc.ave \
    --ngpu 1 \
    --gpu_inference true \
    --nj 1 \
    --inference_nj 1 \
    --use_lm false \
    --lang en \
    --token_type word \
    --max_wav_duration 30 \
    --inference_args "--beam_size 10 --ctc_weight 0.3" \
    --train_set  development \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --bpe_train_text "data/development/text" \
    --lm_train_text "data/development/text" "$@"