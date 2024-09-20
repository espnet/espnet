#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev_clean
test_sets="test_clean"

ssl_opts="--ssl_checkpoint_path exp/kmeans_xues/38epoch.pth --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl --ssl_nlayer 16"
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"
token_list_dir=data/token_list/asr_vocab

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_asr.yaml

./speechlm.sh \
    --task "ssl_asr" \
    --data_name librispeech \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 16 \
    --gpu_inference true \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --token_list_dir ${token_list_dir} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --codec_choice inhouse \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${bpe_opts} ${ssl_opts} "$@"
