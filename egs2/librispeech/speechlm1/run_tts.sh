#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev_clean
test_sets="test_clean"

train_config=conf/train_delay_tts.yaml
inference_config=conf/decode_tts.yaml

token_list_dir=data/token_list/tts_vocab
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

./speechlm.sh \
    --task "tts" \
    --data_name librispeech \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 16 \
    --gpu_inference true \
    --cleaner "tacotron" \
    --g2p "g2p_en_no_space" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --token_list_dir ${token_list_dir} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${codec_opts} \
    "$@"
