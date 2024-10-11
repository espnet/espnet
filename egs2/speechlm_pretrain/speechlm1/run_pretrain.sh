#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_delay_deepspeed_codecssl.yaml
inference_config=conf/decode_inhouse.yaml
inference_model=valid.total_count.ave_5best.till60epoch.pth

token_list_dir=data/token_list/llm_vocab
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B "

train_jsons=""
valid_jsons=""
test_jsons=""

# 1. ASR
asr_train=" \
  dump/raw_codec_ssl_asr_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_asr_librispeech/train_960/data.json \
  dump/raw_codec_ssl_asr_gigaspeech/gigaspeech_train_xl/data.json \
"
asr_valid=" \
  dump/raw_codec_ssl_asr_librispeech/dev_clean/data.json \
"

# 2. TTS
tts_train=" \
  dump/raw_codec_ssl_tts_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_tts_librispeech/train_960/data.json \
  dump/raw_codec_ssl_tts_gigaspeech/gigaspeech_train_xl/data.json \
"
tts_valid=" \
  dump/raw_codec_ssl_tts_librispeech/dev_clean/data.json \
"

# 3. SE
se_train="\
  dump/raw_codec_ssl_se_mls_en/mls_en_train_se_simu/data.json \
  dump/raw_codec_ssl_tse_mls_en/mls_en_train_tse_simu/data.json
"

se_valid="\
  dump/raw_codec_ssl_se_mls_en/mls_en_dev_se_simu/data.json \
  dump/raw_codec_ssl_tse_mls_en/mls_en_dev_tse_simu/data.json \
"

# Audio
audio_train="\
  dump/raw_ag_codecssl_wavcaps/wavcaps_train/data.json \
  dump/raw_aac_codecssl_wavcaps/wavcaps_train/data.json
"
# 3. SSL_ASR
ssl_asr_train=" \
  dump/raw_ssl_asr_mls_en/mls_en_train/data.json \
  dump/raw_ssl_asr_librispeech/train_960/data.json \
  dump/raw_ssl_asr_gigaspeech/gigaspeech_train_xl/data.json \
"
ssl_asr_valid="dump/raw_ssl_asr_librispeech/dev_clean/data.json"

# 4. SSL_TTS
ssl_tts_train=" \
  dump/raw_ssl_tts_mls_en/mls_en_train/data.json \
  dump/raw_ssl_tts_librispeech/train_960/data.json \
  dump/raw_ssl_tts_gigaspeech/gigaspeech_train_xl/data.json \
"
ssl_tts_valid="dump/raw_ssl_tts_librispeech/dev_clean/data.json"

data_combo_name=asr_55k
train_jsons="${asr_train}"
valid_jsons="${asr_valid}"

data_combo_name=tts_55k
train_jsons="${tts_train}"
valid_jsons="${tts_valid}"

# data_combo_name=asr_tts_55k
# train_jsons="${asr_train} ${tts_train}"
# valid_jsons="${asr_valid} ${tts_valid}"

data_combo_name=ssl_asr_55k
train_jsons="${ssl_asr_train}"
valid_jsons="${ssl_asr_valid}"

# data_combo_name=ssl_tts_55k
# train_jsons="${ssl_tts_train}"
# valid_jsons="${ssl_tts_valid}"

data_combo_name=ssl_asr_tts_55k
train_jsons="${ssl_asr_train} ${ssl_tts_train}"
valid_jsons="${ssl_asr_valid} ${ssl_tts_valid}"

data_combo_name=se_tse_45k
train_jsons="${se_train}"
valid_jsons="${se_valid}"

data_combo_name=audio_7k
train_jsons="${audio_train}"
valid_jsons="${asr_valid}"

./speechlm.sh \
    --stage 7 \
    --skip_data_prep true \
    --data_combo_name ${data_combo_name} \
    --fs 16000 \
    --num_nodes 1 \
    --ngpu 4 \
    --nj 200 \
    --inference_nj 8 \
    --nbest 10 \
    --gpu_inference true \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --inference_model ${inference_model} \
    --token_list_dir ${token_list_dir} \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --test_jsons "${test_jsons}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${bpe_opts} "$@"
