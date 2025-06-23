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
n_shift=320
win_length=1280
score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

# discrete related
kmeans_feature="multi/hubert_large_6+wavlm_large_6+wavlm_large_23"
# split with '/', use model_type/layer_index, e.g.:
# 'multi layer': "multi/hubert_l_6+wavlm_l_6+wavlm_l_23";
# 'single layer': "hubert_large_ll60k/6" | "wavlm_large/6" | "wavlm_large/23" | "xls_r_300m/6";
multi_token="hubert_large_ll60k_128_6 wavlm_large_128_6 wavlm_large_128_23"
# concat with ' ', use prepared 'model_type_nclusters_layer_index' tokens
# e.g. "hubert_large_ll60k_128_6_RVQ_0 wavlm_large_128_6_RVQ_0 wavlm_large_128_23_RVQ_0"
mix_type="frame" # frame | sequencee
nclusters=128
RVQ_layers=2
preset_layer=none
preset_token=none

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"

train_config=conf/tuning/train_toksing.yaml
inference_config=conf/tuning/decode.yaml

# text related processing arguments
g2p=none
cleaner=none
pitch_extract=dio

# infer
gpu_inference=true

./svs2.sh \
    --lang zh \
    --stage 1 \
    --stop_stage 9 \
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
    --preset_layer ${preset_layer} \
    --preset_token ${preset_token} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --gpu_inference "${gpu_inference}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    --RVQ_layers "${RVQ_layers}" \
    --kmeans_opts "--batch_bins 4800000" \
    --kmeans_feature "${kmeans_feature}" \
    --multi_token "${multi_token}" \
    --mix_type "${mix_type}" \
    --nclusters "${nclusters}" \
    --RVQ_layers "${RVQ_layers}" \
    --ngpu 1 \
    "$@"
