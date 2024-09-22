#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=16000
fmin=
fmax=
n_fft=
n_shift=
win_length=


train_set=tr_no_dev
valid_set=dev
test_sets="dev test"

train_config=conf/train_delay_tts.yaml
inference_config=conf/decode_tts.yaml

# bpe_opts="--bpemode huggingface --bpemodel allenai/OLMo-1B-hf"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

# pretrained model + task specific tokens
token_list_dir="data/token_list/tts_vocab_ext_phone"
tag="espnet_speechlm_pretrained_tts_ext_phone"

./speechlm_svs.sh \
    --task "svs" \
    --data_name opencpop \
    --fs "${fs}" \
    --ngpu 1 \
    --nj 32 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --token_list_dir "${token_list_dir}" \
    --tag "${tag}" \
    --audio_format "wav" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    --nbest 10 \
    --gpu_inference true \
    ${codec_opts} \
    "$@"