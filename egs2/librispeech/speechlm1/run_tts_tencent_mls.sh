#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_config=conf/train_multiscale_tts.yaml
inference_config=conf/decode_tts.yaml

token_list_dir=data/token_list/tts_vocab
codec_opts="--codec_choice inhouse --dumpdir dump_tencent"

train_jsons="dump_tencent/raw_tts_librispeech/train_960/data.json dump_tencent/raw_tts_mls_en/mls_en_train/data.json"
valid_jsons="dump_tencent/raw_tts_librispeech/dev_clean/data.json dump_tencent/raw_tts_mls_en/mls_en_dev/data.json"
test_jsons="dump_tencent/raw_tts_librispeech/dev_test/data.json"

./speechlm.sh \
    --skip_data_prep true \
    --task "tts" \
    --data_combo_name ls960_mlsen \
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
    --token_list_dir ${token_list_dir} \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --test_jsons "${test_jsons}" \
    ${codec_opts} \
    "$@"
