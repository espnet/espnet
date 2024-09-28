#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_espnet_codec.yaml
inference_model=valid.total_count.best.pth

token_list_dir=data/token_list/asr_vocab
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B "

train_jsons=""
valid_jsons=""
test_jsons=""


data_combo_name=mls_ls_giga_yodas_emilia_asr
train_jsons+=" \
  dump/raw_ssl_asr_mls_en/mls_en_train/data.json \
  dump/raw_ssl_asr_mls_multilingual/mls_multilingual_train/data.json
  dump/raw_ssl_asr_librispeech/train_960/data.json \
  dump/raw_ssl_asr_yodas_manual/train_manual/data.json \
  dump/raw_ssl_asr_yodas_auto2/train_auto_part2/data.json \
  dump/raw_ssl_asr_gigaspeech/gigaspeech_train_xl/data.json \
  dump/raw_ssl_asr_emilia/emilia_en/data.json \
  dump/raw_ssl_asr_yodas_auto1/train_auto_part1/data.json \
"

valid_jsons+=" \
  dump/raw_ssl_asr_librispeech/dev_clean/data.json \
"

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
