#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="full"
task="all"

train_set="w2g_${task}_${lang}_train"
valid_set="w2g_${task}_${lang}_dev"
lm_train_text=data/w2g_${task}_${lang}_train/lm.txt


# number of test sets = number of tasks x number of langs
if [ "$lang" == "full" ]; then
    lang="ady,bej,ers,ixc,kab,kke,kkt,nee,rmn,say,svm,swb,sxg,tvk"
fi
if [ "$task" == "all" ]; then
    task="transcription,surface,underlying,gloss"
fi
test_sets=""
for t in ${task//,/ }; do
    for l in ${lang//,/ }; do
        test_sets+="w2g_${t}_${l}_test "
    done
done


./asr.sh \
    --local_data_opts "--langs ${lang} --tasks ${task}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_tag "${task}_${lang}_xls_r_conformer" \
    --lm_tag "${task}_${lang}_4layer" \
    --nlsyms_txt "data/non_linguistic_symbols.txt" \
    --bpe_nlsyms "data/non_linguistic_symbols.txt" \
    --token_type char \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --min_wav_duration 1 \
    --max_wav_duration 20 \
    --audio_format wav \
    --nbpe 6500 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config conf/tuning/train_xls_r_conformer.yaml \
    --inference_config conf/tuning/decode_transformer.yaml \
    --lm_config conf/tuning/train_lm_4layers.yaml \
    --use_lm true \
    --ngpu 4 \
    --use_text_prev true \
    --bpe_train_text "${lm_train_text}" \
    --lm_train_text "${lm_train_text}" "$@"
