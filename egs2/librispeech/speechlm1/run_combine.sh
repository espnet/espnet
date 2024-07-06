#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_config=conf/train_multiscale.yaml
# train_config=conf/train_multiscale_1b.yaml
# train_config=conf/train_parallel.yaml
# train_config=conf/train_multiscale_delay.yaml
train_config=conf/train_valle_1b.yaml
inference_config=conf/decode_inhouse.yaml
inference_model=valid.total_count.ave_5best.till100epoch.pth
# inference_model=valid.total_count.ave_5best.till75epoch.pth

train_jsons=""
valid_jsons=""
test_jsons=""
data_combo_name=""

use_ls=true
use_giga=true
use_mls_en=true

test_use_ls_7spk=false
test_use_ls=false
test_use_giga=false
test_use_mls_en=false

generate_train_clean_100=false
generate_train_clean_360=false

# download any data repository with: 
#  huggingface-cli download <repo-id> --repo-type dataset --local-dir .

if ${use_ls}; then # <repo-id>: JinchuanTian/tts_librispeech
    train_jsons+="dump/raw_tts_librispeech/train_960/data.json "
    valid_jsons+="dump/raw_tts_librispeech/dev_clean/data.json "
    data_combo_name+="ls_"
fi

if ${use_giga}; then # <repo-id>: JinchuanTian/tts_gigaspeech
    train_jsons+="dump/raw_tts_gigaspeech/gigaspeech_train_xl_spkid/data.json "
    valid_jsons+="dump/raw_tts_gigaspeech/gigaspeech_dev/data.json "
    data_combo_name+="giga_"
fi

if ${use_mls_en}; then # <repo-id>: JinchuanTian/tts_mls_en
    train_jsons+="dump/raw_tts_mls_en/mls_en_train/data.json "
    valid_jsons+="dump/raw_tts_mls_en/mls_en_dev/data.json "
    data_combo_name+="mlsen_"
fi

if ${test_use_ls}; then
    test_jsons+="dump/raw_tts_librispeech/test_clean/data.json "
fi

if ${test_use_ls_7spk}; then
    test_jsons+="dump/raw_tts_librispeech/test_clean_7spk/data.json "
fi

if ${test_use_giga}; then
    test_jsons+="dump/raw_tts_gigaspeech/gigaspeech_test/data.json "
fi

if ${test_use_mls_en}; then
    pass
fi

if ${generate_train_clean_100}; then
    test_jsons+="dump/raw_tts_librispeech/train_clean_100/data.json "
fi

if ${generate_train_clean_360}; then
    test_jsons+="dump/raw_tts_librispeech/train_clean_360/data.json "
fi

test_jsons+="dump/raw_tts_librispeech/dev_clean/data.json "

./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name ${data_combo_name%_} \
    --fs 16000 \
    --ngpu 8 \
    --nj 88 \
    --cleaner "tacotron" \
    --g2p "g2p_en_no_space" \
    --inference_nj 32 \
    --nbest 10 \
    --gpu_inference true \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --inference_model ${inference_model} \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --test_jsons "${test_jsons}" \
    --codec_choice inhouse \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    "$@"
