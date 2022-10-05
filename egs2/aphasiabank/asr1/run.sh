#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="val"
test_sets="test"
include_control=true
include_aphasia_type=false

asr_config=conf/tuning/train_asr_ebranchformer_small.yaml
# asr_config=conf/tuning/train_asr_conformer.yaml
# asr_config=conf/tuning/train_asr_conformer_hubert_ll60k_large.yaml

feats_normalize=global_mvn
if [[ ${asr_config} == *"hubert"* ]]; then
  feats_normalize=utt_mvn # https://github.com/espnet/espnet/issues/4006#issuecomment-1047898558
fi

asr_tag="$(basename "${asr_config}" .yaml)"
if "${include_aphasia_type}"; then
  asr_tag+="_with_aphasia_type"
fi

inference_config=conf/decode.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false

./asr.sh \
  --lang en \
  --inference_nj 100 \
  --ngpu 1 \
  --audio_format wav \
  --feats_type raw \
  --token_type char \
  --use_lm ${use_lm} \
  --lm_config "${lm_config}" \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --feats_normalize ${feats_normalize} \
  --local_data_opts "--include_control ${include_control} --include_aphasia_type ${include_aphasia_type}" \
  --asr_tag "${asr_tag}" \
  --lm_train_text "data/${train_set}/text" "$@"

# TODO: post-processing inference output to remove aphasia type markers
