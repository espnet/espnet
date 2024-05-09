#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train-clean-960
valid_set=dev-clean
test_sets="dev-clean test-clean"

train_config=conf/train_multiscale.yaml
inference_config=conf/decode_encodec.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

./speechlm.sh \
    --stage 9 --stop_stage 9 \
    --task "tts" \
    --fs 24000 \
    --ngpu 4 \
    --nj 32 \
    --inference_nj 1 --gpu_inference true \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --local_data_opts "${local_data_opts}" \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir dump_encodec \
    --data_tag train-clean-960_tts_encodec \
    --codec_choice EnCodec \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    "$@"