#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=val
test_sets=test

# train_config=conf/train_valle.yaml
train_config=conf/train_delay_tts.yaml
# train_config=conf/train_multiscale.yaml
inference_config=conf/decode_tts.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
# local_data_opts="--trim_all_silence false" # trim all silence in the audio
local_data_opts="--model_conf large --task visual_tts --data_name lrs2 --train_set ${train_set} --valid_set ${valid_set} --test_sets ${test_sets}"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

# nj: 分段的数量
./speechlm.sh \
    --stage 6 \
    --stop_stage 9 \
    --task "visual_tts" \
    --data_name lrs2 \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --local_data_opts "${local_data_opts}" \
    --audio_format "wav" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 0.1 \
    --max_wav_duration 30.0 \
    ${codec_opts} "$@"
