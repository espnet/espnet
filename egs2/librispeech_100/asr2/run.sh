#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="espnet_hubert/16"  # use model_type/layer_index
nclusters=5000
model_path="/scratch/bbjs/chen26/espnet_ssl/egs2/librispeech/ssl1/exp_900k/hubert_iter2_train_ssl_espnethubert_ebf_ce_large_p3_raw/38epoch.pth"

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set="dev_aishell"
train_dev="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --kmeans_opts "--batch_bins 4800000 --nj 1 --hubert_dir_path ${model_path}" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --audio_format flac.ark \
    --ngpu 1 \
    --stage 5 \
    --stop_stage 5 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "dump/raw/${train_set}_sp/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "dump/raw/${train_set}_sp/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "dump/raw/${train_set}_sp/text.${tgt_case}.${tgt_lang}" "$@"
