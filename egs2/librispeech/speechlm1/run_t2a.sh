#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=test_clean
valid_set=dev_clean
test_sets=""

train_config=conf/train_delay_t2a.yaml
inference_config=conf/decode_tts.yaml

token_list_dir=data/token_list/t2a_vocab
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"
text_emb_opts="--textlm_hf_model_tag google-t5/t5-large --textlm_max_words 500"

./speechlm.sh \
    --task "text2audio" \
    --data_name librispeech \
    --fs 16000 \
    --ngpu 1 \
    --nj 1 \
    --inference_nj 1 \
    --nbest 10 \
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
    ${text_emb_opts} ${codec_opts} \
    "$@"
