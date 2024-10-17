#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="en"
data_split="full" # one of full 1h 10h
local_data_opts="--lang ${lang} --data_split ${data_split}"

train_set="mls_${lang}_train"
valid_set="mls_${lang}_dev"
test_sets="mls_${lang}_test"

train_config=conf/train_delay_tts.yaml
inference_config=conf/decode_tts.yaml

token_list_dir=data/token_list/asr_vocab
ssl_opts="\
  --ssl_choice espnet_hubert \
  --ssl_checkpoint_path exp/kmeans_xues/38epoch.pth \
  --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl \
  --ssl_nlayer 16 \
  --dumpdir dump_xeus \
"
subword_opts="\
  --subword_choice sentencepiece \
  --nbpe 1000 \
"

./speechlm.sh \
    --task "ssl_asr" \
    --data_name mls_en \
    --fs 16000 \
    --ngpu 1 \
    --nj 88 \
    --inference_nj 88 \
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
    ${ssl_opts} ${subword_opts} \
    "$@"
