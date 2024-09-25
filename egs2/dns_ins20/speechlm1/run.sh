#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16000
extra_annotations=UNK

train_set=tr_synthetic
valid_set=cv_synthetic
test_sets="tt_synthetic_no_reverb tt_synthetic_with_reverb"

token_list_dir=../../librispeech/speechlm1/data/token_list/tts_vocab
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

./speechlm.sh \
    --task "se" \
    --data_name "dnsins20" \
    --fs $sample_rate \
    --nj 16 \
    --inference_nj 16 \
    --audio_format "wav.ark" \
    --train_config conf/tuning/train_delay_se_init.yaml \
    --inference_config conf/tuning/decode_se_topk.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_model valid.acc.ave.pth \
    --max_wav_duration 31 \
    --token_list_dir ${token_list_dir} \
    $codec_opts \
    "$@"
