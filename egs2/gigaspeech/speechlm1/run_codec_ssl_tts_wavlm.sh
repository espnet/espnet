#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="test"

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_asr.yaml

dumpdir=dump_wavlm
ssl_opts="--ssl_choice s3prl --ssl_feature_type wavlm_large --ssl_nlayer 21 --ssl_kmeans_path exp/kmeans_wavlm/wavlm_large_21_2000clusters/km_2000.mdl --ssl_batch_bins 19200000"
codec_opts="--codec_choice inhouse"
bpe_opts="--subword_choice sentencepiece --nbpe 5000"


./speechlm.sh \
    --task "codec_ssl_tts" \
    --data_name gigaspeech \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    --dumpdir ${dumpdir} \
    ${ssl_opts} ${codec_opts} ${bpe_opts} \
    "$@"
