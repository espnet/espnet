#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=te # te ta gu

train_set=train_${lang}
train_dev=dev_${lang}
test_set="${train_dev} test_${lang}"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decoder_asr.yaml

if [[ "zh" == *"${lang}"* ]]; then # placeholder for optimal bpe when lang=te
  nbpe=2500
elif [[ "fr" == *"${lang}"* ]]; then # placeholder for optimal bpe when lang=ta
  nbpe=350
elif [[ "es" == *"${lang}"* ]]; then # placeholder for optimal bpe when lang=gu
  nbpe=235
else
  nbpe=150
fi


./asr.sh \
    --ngpu 1 \
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
    --inference_asr_model valid.acc.ave.pth\
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --min_wav_duration 0.5 \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
