#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=16000
fmin=80
fmax=7600
n_fft=2048
n_shift=160
win_length=1200
score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

kmeans_feature="mfcc"
#"wavlm_large/21"  # use model_type/layer_index
nclusters=1024

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"

train_config=conf/tuning/train_naive_rnn_dp.yaml
inference_config=conf/tuning/decode_rnn.yaml
expdir=exp/test

src_nbpe=5000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

# text related processing arguments
g2p=pyopenjtalk
cleaner=none
pitch_extract=None

./svs2.sh \
    --lang jp \
    --stage 1 \
    --stop_stage 3 \
    --local_data_opts "--stage 0" \
    --feats_type raw \
    --pitch_extract "${pitch_extract}" \
    --fs "${fs}" \
    --fmax "${fmax}" \
    --fmin "${fmin}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    --svs_exp "${expdir}" \
    --kmeans_opts "--batch_bins 4800000" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "char" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "char" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --inference_config "${inference_config}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang} data/local/other_text/text" "$@"
