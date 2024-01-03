#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="wavlm_large/21"  # use model_type/layer_index
nclusters=2000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set="train_small"
train_dev="dev"
test_sets="dev test_clean test_other"

asr_config_ctc=conf/tuning/train_discrete_asr_e_branchformer_mistral02_ctc.yaml
asr_config_aed=conf/tuning/train_discrete_asr_e_branchformer_mistral02_aed.yaml
inference_config=conf/decode_hf.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=3000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

embed_tokens_path=data/mistral7b-instruct02_embed_tokens.pth

local/export_hf_embed_tokens.py \
    mistralai/Mistral-7B-Instruct-v0.2 \
    ${embed_tokens_path}

./asr2.sh \
    --kmeans_feature "${kmeans_feature}" \
    --kmeans_opts "--batch_bins 4800000 --nj 1" \
    --use_lm false \
    --use_ngram false \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --audio_format "flac.ark" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "hugging_face" \
    --hugging_face_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config_ctc}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --pretrained_model ${embed_tokens_path} \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --stop_stage 13

./asr2.sh \
    --kmeans_feature "${kmeans_feature}" \
    --kmeans_opts "--batch_bins 4800000 --nj 1" \
    --use_lm false \
    --use_ngram false \
    --nclusters "${nclusters}" \
    --ngpu 4 \
    --gpu_inference true \
    --inference_nj 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --audio_format "flac.ark" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "hugging_face" \
    --hugging_face_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config_aed}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --pretrained_model "exp/asr_train_discrete_asr_e_branchformer_mistral02_ctc_raw_wavlm_large_21_km2000_bpe_rm6000_hugging_face_ts_mistralai-Mistral-7B-Instruct-v0.2_sp/valid.cer_ctc.ave_10best.pth:::ctc" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --stage 13
