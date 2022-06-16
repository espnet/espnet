#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=all # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk

train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set="${train_dev} test_$(echo ${lang} | tr - _)"

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/tuning/train_asr_transformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

if [[ "zh" == *"${lang}"* ]]; then
  nbpe=2500
elif [[ "fr" == *"${lang}"* ]]; then
  nbpe=350
elif [[ "es" == *"${lang}"* ]]; then
  nbpe=235
else
  nbpe=150
fi

if [[ "all" == *"${lang}"* ]]; then
  ./asr.sh \
      --lang "${lang}" \
      --local_data_opts "--stage 0 --lang ${lang} --nlsyms_txt ${nlsyms_txt}" \
      --use_lm true \
      --lm_config "${lm_config}" \
      --token_type bpe \
      --nbpe $nbpe \
      --bpe_nlsyms "{$nlsyms_txt}" \
      --feats_type raw \
      --speed_perturb_factors "0.9 1.0 1.1" \
      --asr_config "${asr_config}" \
      --inference_config "${inference_config}" \
      --train_set "${train_set}" \
      --valid_set "${train_dev}" \
      --test_sets "${test_set}" \
      --bpe_train_text "data/${train_set}/text" \
      --lm_train_text "data/${train_set}/text" "$@" \
      --local_score_opts "--score_lang_id true" "$@"
else
  ./asr.sh \
      --lang "${lang}" \
      --local_data_opts "--lang ${lang}" \
      --use_lm true \
      --lm_config "${lm_config}" \
      --token_type bpe \
      --nbpe $nbpe \
      --feats_type raw \
      --speed_perturb_factors "0.9 1.0 1.1" \
      --asr_config "${asr_config}" \
      --inference_config "${inference_config}" \
      --train_set "${train_set}" \
      --valid_set "${train_dev}" \
      --test_sets "${test_set}" \
      --bpe_train_text "data/${train_set}/text" \
      --lm_train_text "data/${train_set}/text" "$@" 
fi
