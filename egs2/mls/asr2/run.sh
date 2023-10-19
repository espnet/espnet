#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

kmeans_feature=wavlm_large/21  # use model_type/layer_index
nclusters=1000

src_lang=$(echo "${kmeans_feature}_full_km${nclusters}" | tr "/" "_")
tgt_lang=multi

lang="all" # one of all es en fr nl it pt pl de
data_split="10h" # one of full 1h 10h

train_set="mls_${lang}_train"
valid_set="mls_${lang}_dev"
lm_train_text=data/${lang}_lm_train.txt

if [ "$lang" == "all" ]; then
    for ln in es en fr nl it pt pl de; do
        test_sets+="mls_${ln}_test "
    done
else
    test_sets="mls_${lang}_test"
fi

asr_config=conf/train_discrete_asr_e_branchformer1.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=3000
tgt_nbpe=150

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --local_data_opts "--lang ${lang} --data_split ${data_split}" \
    --portion 1.0 \
    --kmeans_opts "--batch_bins 4800000" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"
