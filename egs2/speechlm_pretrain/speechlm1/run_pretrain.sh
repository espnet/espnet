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
  dump/raw_codec_ssl_asr_yodas_auto1/train_auto_part1/data.json \
  dump/raw_codec_ssl_asr_yodas_auto2/train_auto_part2/data.json \
  dump/raw_codec_ssl_asr_yodas_manual/train_manual/data.json \
  dump/raw_codec_ssl_asr_emilia/emilia_en/data.json \
  dump/raw_codec_ssl_asr_mls_multilingual/mls_multilingual_train/data.json \
"
asr_valid=" \
  dump/raw_codec_ssl_asr_librispeech/dev_clean/data.json \
"

# 2. TTS
tts_train=" \
  dump/raw_codec_ssl_tts_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_tts_librispeech/train_960/data.json \
  dump/raw_codec_ssl_tts_gigaspeech/gigaspeech_train_xl/data.json \
  dump/raw_codec_ssl_tts_yodas_auto1/train_auto_part1/data.json \
  dump/raw_codec_ssl_tts_yodas_auto2/train_auto_part2/data.json \
  dump/raw_codec_ssl_tts_yodas_manual/train_manual/data.json \
  dump/raw_codec_ssl_tts_emilia/emilia_en/data.json \
"
tts_valid=" \
  dump/raw_codec_ssl_tts_librispeech/dev_clean/data.json \
"

data_combo_name=asr_180k
train_jsons="${asr_train}"
valid_jsons="${asr_valid}"

data_combo_name=tts_180k
train_jsons="${tts_train}"
valid_jsons="${tts_valid}"

data_combo_name=asr_tts_180k
train_jsons="${asr_train} ${tts_train}"
valid_jsons="${asr_valid} ${tts_valid}"

./speechlm.sh \
    --stage 7 \
    --skip_data_prep true \
    --data_combo_name ${data_combo_name} \
    --fs 16000 \
    --num_nodes 1 \
    --ngpu 8 \
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
