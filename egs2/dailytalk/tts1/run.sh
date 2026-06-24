#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# DailyTalk audio is 44.1 kHz. Downsample to the LJSpeech 22.05 kHz baseline
# to keep the first end-to-end experiment reasonably small.
fs=22050
n_fft=1024
n_shift=256

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

train_config=conf/train.yaml
inference_config=conf/decode.yaml
tts_task=gan_tts
inference_model=train.total_count.ave.pth
g2p=g2p_en_no_space

./tts.sh \
    --lang en \
    --feats_type raw \
    --audio_format flac \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner tacotron \
    --g2p "${g2p}" \
    --use_sid true \
    --tts_task "${tts_task}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_model "${inference_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    "$@"
