#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

duration=10min

multilingual=false
lid=false
single_lang=eng

if "${multilingual}"; then
    train_set=train_${duration}
    train_dev=dev_${duration}
    test_set="${train_dev} test_${duration}"
else
    train_set=train_${duration}_${single_lang}
    train_dev=dev_${duration}_${single_lang}
    test_set="${train_dev} test_${duration}_${single_lang}"
fi

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/tuning/train_asr_fbank_single.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --ngpu 1 \
    --local_data_opts "--duration ${duration} --lid ${lid} --multilingual ${multilingual} --single_lang ${single_lang} --nlsyms_txt ${nlsyms_txt}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type char \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"

