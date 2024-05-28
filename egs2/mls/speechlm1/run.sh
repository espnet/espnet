#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="en" 
data_split="full" # one of full 1h 10h

train_set="mls_${lang}_train"
valid_set="mls_${lang}_dev"
test_sets=""

train_config=conf/train_valle.yaml
inference_config=conf/decode_encodec.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en

# Note(Jinchuan): We only select audio range from 3s to 30s since:
#                 (1) The speech prompt is 3s
#                 (2) We limit the longest audio to 30s to avoid
#                     some corner cases in memeory
./speechlm.sh \
    --local_data_opts "--lang ${lang} --data_split ${data_split}" \
    --task "tts" \
    --fs 16000 \
    --ngpu 4 \
    --nj 64 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --codec_choice inhouse \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    "$@"
