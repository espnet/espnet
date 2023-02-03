#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


multilingual=false
lid=false
nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/tuning/train_asr_fbank_single.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml



for duration in 10min 1h; do
    echo ${duration}
    for single_lang in eng1 eng2 eng3 fra1 fra2 deu1 deu2 rus swa swe jpn cmn xty ; do
        echo ${single_lang}
        train_set=train_${duration}_${single_lang}
        train_dev=dev_10min_${single_lang}
        test_set="${train_dev} test_10min_${single_lang}"
        lang=${single_lang}
        asr_tag="$(basename "${asr_config}" .yaml)_${single_lang}_${duration}"

        ./asr.sh \
            --ngpu 1 \
            --stage 1 \
            --stop_stage 13 \
            --lang ${lang} \
            --nj 4 \
            --inference_nj 4 \
            --inference_asr_model "valid.loss.ave.pth" \
            --local_data_opts "--duration ${duration} --lid ${lid} --multilingual ${multilingual} --single_lang ${single_lang} --nlsyms_txt ${nlsyms_txt}" \
            --use_lm false \
            --lm_config "${lm_config}" \
            --token_type char \
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
            --lm_train_text "data/${train_set}/text" "$@"
    done
done