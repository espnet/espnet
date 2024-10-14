#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=valid
test_sets="test"

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_tts.yaml

token_list_dir=data/token_list/asr_vocab
ssl_opts="\
  --ssl_choice espnet_hubert \
  --ssl_checkpoint_path exp/kmeans_xues/38epoch.pth \
  --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl \
  --ssl_nlayer 16 \
  --dumpdir dump_voxtlm \
"
subword_opts="\
  --subword_choice sentencepiece \
  --nbpe 1000 \
"

./speechlm.sh \
    --task "ssl_asr" \
    --data_name librispeech \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
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
