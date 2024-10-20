#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_delay_tts.yaml
inference_config=conf/decode_inhouse.yaml
inference_model=valid.total_count.ave_5best.till60epoch.pth

token_list_dir=data/token_list/llm_vocab

# se
data_combo_name=mlsen_se
train_jsons="dump/raw_codec_ssl_se_mls_en/mls_en_train_se_simu/data.json"
valid_jsons="dump/raw_codec_ssl_se_mls_en/mls_en_dev_se_simu/data.json"
test_jsons="dump/raw_codec_ssl_se_mls_en/mls_en_test_se_simu/data.json"

# TSE
data_combo_name=mlsen_tse
train_jsons="dump/raw_codec_ssl_tse_mls_en/mls_en_train_tse_simu/data.json"
valid_jsons="dump/raw_codec_ssl_tse_mls_en/mls_en_dev_tse_simu/data.json"
test_jsons="dump/raw_codec_ssl_tse_mls_en/mls_en_test_tse_simu/data.json"


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
    "$@"
