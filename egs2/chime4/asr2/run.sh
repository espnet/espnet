#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

kmeans_feature="wavlm_large/21"  # use model_type/layer_index
nclusters=1000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="dt05_real_beamformit_5mics et05_real_beamformit_5mics "

asr_config=conf/tuning/train_discrete_asr_e_branchformer_e12_mlp1024_linear1024_macaron_lr1e-4_warmup25k_conv1d1.yaml
inference_config=conf/decode_asr.yaml
lm_config=

speed_perturb_factors="0.9 1.0 1.1"

gpu_inference=false

src_nbpe=2000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

use_word_lm=false
word_vocab_size=65000

./asr2.sh                                   \
    --kmeans_feature "${kmeans_feature}" \
    --use_lm false \
    --use_ngram false \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --audio_format "flac.ark" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "char" \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
