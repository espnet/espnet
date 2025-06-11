#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

### Before You Start ###
# (1) Install the extra dependency of ESPnet-SpeechLM following the link:
#     Check the "Environment" section of <espnet>/egs2/Template/speechlm1/README.md
# (2) Download the pre-trained tokenizers:
#     huggingface-cli download --repo-type model --local-dir . JinchuanTian/OpusLM_v0_1.7B_NAACL_Demo

train_set="test_clean"
valid_set= # "dev"
test_sets="test_clean"

task="codec_ssl_asr" # codec_ssl_asr or codec_ssl_tts

if [ ${task} == "codec_ssl_asr" ]; then
    # test_sets="test_clean test_other"
    inference_config=conf/decode_asr.yaml
    nbest=1
elif [ ${task} == "codec_ssl_tts" ]; then
    test_sets="test_clean"
    inference_config=conf/decode_tts.yaml
    nbest=10
else
    echo "This recipe only support codec_ssl_asr and codec_ssl_tts task"
fi

train_config=conf/train_delay_smollm_360m.yaml

# Tokenizer setups
token_list_dir=data/token_list/llm_vocab2
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-360M"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch"
ssl_opts="--ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 20000000"

./speechlm.sh \
    --skip_data_prep false \
    --data_name librispeech \
    --task ${task} \
    --fs 16000 \
    --ngpu 2 \
    --nj 1 \
    --inference_nj 1 \
    --nbest ${nbest} \
    --gpu_inference true \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir dump \
    --token_list_dir ${token_list_dir} \
    ${bpe_opts} ${codec_opts} ${ssl_opts} \
    "$@"
