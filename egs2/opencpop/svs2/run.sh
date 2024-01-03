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
win_length=1200
score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

kmeans_feature=""
# "wavlm_large/6" | "encodec/1" | "xls_r_300m/6", use model_type/layer_index
multi_token=""
# "wav_large_1024_6 xls_r_330m_1024_6", use prepared 'model_type_nclusters_layer_index' tokens
mix_type="frame"
# frame | sequencee
nclusters=1024

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set=tr_no_dev
valid_set=dev
test_sets="dev test"

train_config=conf/tuning/train_xiaoice.yaml
inference_config=conf/tuning/decode_rnn.yaml

# text related processing arguments
g2p=none
cleaner=none
pitch_extract=dio

./svs2.sh \
    --lang zh \
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
    --kmeans_opts "--batch_bins 4800000" \
    --kmeans_feature "${kmeans_feature}" \
    --multi_token "${multi_token}" \
    --mix_type "${mix_type}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    "$@"
