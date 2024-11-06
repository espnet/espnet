#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

tasks="codec_ssl_tts codec_ssl_asr"
tasks="codec_ssl_tts"

train_jsons=""
valid_jsons=""
test_jsons=""

for task in ${tasks}; do
    train_jsons+="dump_wavlm/raw_${task}_librispeech/train_960/data.json dump_wavlm/raw_${task}_mls_en/mls_en_train/data.json "
    valid_jsons+="dump_wavlm/raw_${task}_librispeech/dev/data.json "
    test_jsons+="dump_wavlm/raw_${task}_librispeech/test_clean/data.json"
done

train_config=conf/train_delay_smollm_360m.yaml
inference_config=conf/decode_tts_tencent.yaml

token_list_dir=data/token_list/llm_vocab # use lllm vocab
ssl_opts="--ssl_choice s3prl --ssl_feature_type wavlm_large --ssl_nlayer 21 --ssl_kmeans_path exp/kmeans_wavlm/wavlm_large_21_2000clusters/km_2000.mdl --ssl_batch_bins 19200000"
codec_opts="--codec_choice inhouse"
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"

./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name ls960_mlsen \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --test_jsons "${test_jsons}" \
    --dumpdir dump_wavlm \
    ${bpe_opts} ${ssl_opts} ${codec_opts}\
    "$@"
