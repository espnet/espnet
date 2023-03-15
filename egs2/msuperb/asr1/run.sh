#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

duration=10min

multilingual=true
lid=false
only_lid=true
single_lang=xty
stage=1
nj=4

token_type=char
if "${multilingual}"; then
    if "${only_lid}"; then
        suffix="_only_lid"
        token_type=word
    else
        if "${lid}"; then
            suffix="_lid"
        else
            suffix=""
        fi
    fi
    train_set=train_${duration}${suffix}
    train_dev=dev_${duration}${suffix}
    test_set="${train_dev} test_${duration}${suffix}"
    lang="multilingual"
else
    train_set=train_${duration}_${single_lang}
    train_dev=dev_${duration}_${single_lang}
    test_set="${train_dev} test_${duration}_${single_lang}"
    lang=${single_lang}
fi

nlsyms_txt=data/local/nlsyms.txt
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml
asr_config=conf/tuning/train_asr_fbank.yaml
asr_tag="$(basename "${asr_config}" .yaml)_${lang}_${duration}"

./asr.sh \
    --ngpu 1 \
    --stage ${stage} \
    --lang ${lang} \
    --nj ${nj} \
    --inference_nj ${nj} \
    --inference_asr_model valid.loss.ave.pth \
    --local_data_opts "--duration ${duration} --lid ${lid} --only_lid ${only_lid} --multilingual ${multilingual} --single_lang ${single_lang} --nlsyms_txt ${nlsyms_txt}" \
    --nlsyms_txt ${nlsyms_txt} \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type ${token_type} \
    --feats_type raw \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --asr_tag "${asr_tag}" \
    --asr_stats_dir exp/asr_stats_${lang}_${duration} \
    --lm_train_text "data/${train_set}/text" "$@" \
    --local_score_opts "${lid} ${only_lid}"
