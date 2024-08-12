#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train-960
valid_set=dev-clean
test_sets="test-clean"

train_config=conf/train_valle.yaml
# train_config=conf/train_delay.yaml
# train_config=conf/train_multiscale.yaml
inference_config=conf/decode_espnet_codec.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"


./speechlm.sh \
    --task "tts" \
    --data_name libritts \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --local_data_opts "${local_data_opts}" \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 1.0 \
    --max_wav_duration 30.0 \
    ${codec_opts} "$@"
