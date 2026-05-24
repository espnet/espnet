#!/usr/bin/env bash
# Re-decode the EXISTING e_branchformer_scratch ASR model with use_lm=false,
# as a controlled ablation against the LM-rescored run. Same trained model,
# same SP-tripled data, only difference is the absence of LM rescoring.
#
# Re-using --asr_tag e_branchformer_scratch points asr.sh at the existing
# exp/asr_e_branchformer_scratch/ checkpoint dir. The decode output dir is
# distinct because --use_lm false changes its name to a non-LM-prefixed form.
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr_branchformer.yaml   # no-LM decode config
speed_perturb_factors="0.9 1.0 1.1"
asr_tag=e_branchformer_scratch

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 2 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm false \
    --use_word_lm false \
    --asr_config "${asr_config}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
