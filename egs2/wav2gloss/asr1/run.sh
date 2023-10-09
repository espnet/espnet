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

asr_config="conf/tuning/train_xls_r_conformer.yaml"
asr_tag="${task}_${lang}_xls_r_conformer"

lm_config="conf/tuning/train_lm_4layers.yaml"
lm_tag="${task}_${lang}_4layer"

# number of test sets = number of tasks x number of langs
if [ "$lang" == "full" ]; then
    lang="adyg1241,ainu1240,apah1238,arap1274,arta1239,balk1252,beja1238,bora1263,dolg1241,even1259,goro1270,jeju1234,kaby1243,kach1280,kaka1265,kama1378,kara1499,koii1238,komn1238,mand1415,nngg1234,nort2641,pnar1238,port1286,ruul1235,sanz1248,savo1255,selk1253,slav1254,sout2856,sumb1241,sumi1235,taba1259,taul1251,tehr1242,teop1238,texi1237,tond1251,trin1278,vera1241"
fi
if [ "$task" == "all" ]; then
    task="transcription,underlying,gloss,translation"
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
    --nlsyms_txt "data/non_linguistic_symbols.txt" \
    --bpe_nlsyms "data/non_linguistic_symbols.txt" \
    --token_type char \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --min_wav_duration 1 \
    --max_wav_duration 20 \
    --audio_format wav \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config ${asr_config} \
    --asr_tag ${asr_tag} \
    --lm_config ${lm_config} \
    --lm_tag ${lm_tag} \
    --inference_config conf/tuning/decode_transformer.yaml \
    --use_lm true \
    --ngpu 4 \
    --gpu_inference true \
    --use_text_prev true \
    --bpe_train_text "${lm_train_text}" \
    --lm_train_text "${lm_train_text}" "$@"
