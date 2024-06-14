#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train-clean-460
valid_set=dev-clean
test_sets="test-clean"

train_config=conf/train_valle.yaml
# train_config=conf/train_multiscale.yaml
inference_config=conf/decode_espnet.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

# Note(Jinchuan): We only select audio range from 3s to 30s since:
#                 (1) The speech prompt is 3s
#                 (2) We limit the longest audio to 30s to avoid
#                     some corner cases in memeory
./speechlm.sh \
    --task "tts" \
    --data_name libritts \
    --fs 24000 \
    --ngpu 4 \
    --nj 64 \
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
    --codec_choice ESPnet \
    --codec_checkpoint_path espnet_codec/24khz_soundstream/train.total_count.best.pth \
    --codec_config_path espnet_codec/24khz_soundstream/config.yaml \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    "$@"
