#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

kmeans_feature="wavlm_large/21"  # use model_type/layer_index
nclusters=2000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=jp

fs=16000
opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set="tr_no_dev"
train_dev="dev"
test_sets="dev eval1"

asr_config=conf/train_discrete_asr_e_branchformer1.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --kmeans_opts "--batch_bins 2000000 --nj 1 --num_threads 4" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --nj 8 \
    --ngpu 2 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "char" \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --fs ${fs} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm false \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    ${opts} "$@"
