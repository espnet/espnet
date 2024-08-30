#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_delay.yaml
inference_config=conf/decode_inhouse.yaml
inference_model=valid.total_count.ave_5best.till60epoch.pth

token_list_dir=data/token_list/llm_vocab
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B "

train_jsons=""
valid_jsons=""
test_jsons=""

# As of Aug 29, MLS_en + LibriSpeech + Yodas + GigaSpeech
data_combo_name=combo_aug29
# 1. ASR
train_jsons+=" \
  dump/raw_codec_ssl_asr_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_asr_librispeech/train_960/data.json \
  dump/raw_codec_ssl_asr_yodas_manual/train_manual/data.json \
  dump/raw_codec_ssl_asr_yodas_auto2/train_auto_part2/data.json \
  dump/raw_codec_ssl_asr_yodas_auto1/train_auto_part1/data.json \
  dump/raw_codec_ssl_asr_gigaspeech/gigaspeech_train_xl/data.json \
"
valid_jsons+=" \
  dump/raw_codec_ssl_asr_mls_en/mls_en_dev/data.json \
  dump/raw_codec_ssl_asr_librispeech/dev_clean/data.json \
  dump/raw_codec_ssl_asr_gigaspeech/gigaspeech_dev/data.json \
"

# 2. TTS
train_jsons+=" \
  dump/raw_codec_ssl_tts_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_tts_librispeech/train_960/data.json \
  dump/raw_codec_ssl_tts_yodas_manual/train_manual/data.json \
  dump/raw_codec_ssl_tts_yodas_auto1/train_auto_part1/data.json \
  dump/raw_codec_ssl_tts_yodas_auto2/train_auto_part2/data.json \
  dump/raw_codec_ssl_tts_gigaspeech/gigaspeech_train_xl/data.json \
"
valid_jsons+=" \
  dump/raw_codec_ssl_tts_mls_en/mls_en_dev/data.json \
  dump/raw_codec_ssl_tts_librispeech/dev_clean/data.json \
  dump/raw_codec_ssl_tts_gigaspeech/gigaspeech_dev/data.json \
"

# 3. Audio Auto-Regressive
train_jsons+=" \
  dump/raw_codec_ssl_audiolm_mls_en/mls_en_train/data.json \
  dump/raw_codec_ssl_audiolm_librispeech/train_960/data.json
  dump/raw_codec_ssl_audiolm_yodas_auto1/train_auto_part1/data.json \
  dump/raw_codec_ssl_audiolm_yodas_auto2/train_auto_part2/data.json \
  dump/raw_codec_ssl_audiolm_yodas_manual/train_manual/data.json \
  dump/raw_codec_ssl_audiolm_gigaspeech/gigaspeech_train_xl/data.json \
"
valid_jsons+=" \
  dump/raw_codec_ssl_audiolm_mls_en/mls_en_test/data.json \
  dump/raw_codec_ssl_audiolm_librispeech/dev_clean/data.json \
  dump/raw_codec_ssl_audiolm_gigaspeech/gigaspeech_dev/data.json \
"

echo ${data_combo_name}
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
