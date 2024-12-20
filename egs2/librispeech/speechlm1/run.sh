#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_jsons=""
valid_jsons=""
test_jsons=""

# (1) Codec_SSL_TTS
train_jsons+="dump/raw_codec_ssl_tts_librispeech/train_960/data.json "
train_jsons+="dump/raw_codec_ssl_tts_mls_en/mls_en_train/data.json "
train_jsons+="dump/raw_codec_ssl_tts_mls_multilingual/mls_multilingual_train/data.json "
train_jsons+="dump/raw_codec_ssl_tts_emilia/emilia_en/data.json "
train_jsons+="dump/raw_codec_ssl_tts_yodas/train_yodas/data.json "
valid_jsons+="dump/raw_codec_ssl_tts_librispeech/dev/data.json " 

# (2) Codec_SSL_ASR
train_jsons+="dump/raw_codec_ssl_asr_librispeech/train_960/data.json "
train_jsons+="dump/raw_codec_ssl_asr_mls_en/mls_en_train/data.json "
train_jsons+="dump/raw_codec_ssl_asr_mls_multilingual/mls_multilingual_train/data.json "
train_jsons+="dump/raw_codec_ssl_asr_emilia/emilia_en/data.json "
train_jsons+="dump/raw_codec_ssl_asr_yodas/train_yodas/data.json "
train_jsons+="dump/raw_codec_ssl_asr_owsm_en/owsm_train/data.json "
valid_jsons+="dump/raw_codec_ssl_asr_librispeech/dev/data.json "

# (3) TextLM
# train_jsons+="dump/raw_textlm_librispeech/librispeech_text/data.json "
# for x in `ls dump/raw_textlm_cc_whole`; do
#     train_jsons+="dump/raw_textlm_cc_whole/${x}/data.json "
# done
# for x in `ls dump/raw_textlm_cc_whole_llama`; do
#     train_jsons+="dump/raw_textlm_cc_whole_llama/${x}/data.json "
# done
for x in `ls dump/raw_textlm_cc_whole_qwen`; do
    train_jsons+="dump/raw_textlm_cc_whole_qwen/${x}/data.json "
done

# (4) AudioLM
train_jsons+="dump/raw_codec_ssl_audiolm_librispeech/train_960/data.json "
train_jsons+="dump/raw_codec_ssl_audiolm_mls_en/mls_en_train/data.json "
train_jsons+="dump/raw_codec_ssl_audiolm_mls_multilingual/mls_multilingual_train/data.json "
train_jsons+="dump/raw_codec_ssl_audiolm_emilia/emilia_en/data.json "
train_jsons+="dump/raw_codec_ssl_audiolm_yodas/train_yodas/data.json "
train_jsons+="dump/raw_codec_ssl_audiolm_owsm_en/owsm_train/data.json "


# Test sets
# test_jsons+="dump/raw_codec_ssl_asr_librispeech/test_clean/data.json "
# test_jsons+="dump/raw_codec_ssl_tts_librispeech/test_clean/data.json "

train_config=conf/train_delay_smollm_360m.yaml
inference_config=conf/decode_asr.yaml

token_list_dir=data/token_list/llm_vocab2 # use lllm vocab
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"

./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name ls960_mlsen \
    --fs 16000 \
    --ngpu 4 \
    --nj 200 \
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
    --dumpdir dump \
    ${bpe_opts} \
    "$@"
