#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000 # 44100 24000 yw change the default frequency
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 44100 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi # opts="--audio_format wav " # flac for saving storage space, but require further processing afterwards.

train_config=conf/train_fastspeech2.yaml
inference_config=conf/decode_fastspeech2.yaml

# train_set=train_no_dev
# valid_set=dev
# test_sets="dev test"
# g2p=pypinyin_g2p_phone
# Input: 卡尔普陪外孙玩滑梯
# pypinyin_g2p: ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1
# pypinyin_g2p_phone: k a3 er3 p u3 p ei2 uai4 s un1 uan2 h ua2 t i1

# if you want to use officially provided phoneme text (better for the quality)
train_set=train_no_dev_phn
valid_set=dev_phn
test_sets="test_phn"
g2p=none

vocoder_file="vocoder/checkpoint-180000steps_cn.pkl" # vocoder/vocoder.pkl

./tts2.sh \
    --lang zh \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --src_token_type phn \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_spk_embed true \
    --teacher_dumpdir exp/tts_train_teacher_raw_phn_none/decode_teacher_use_teacher_forcingtrue_train.loss.ave \
    --tts2_stats_dir exp/tts_train_teacher_raw_phn_none/decode_teacher_use_teacher_forcingtrue_train.loss.ave/stats \
    --tts2_exp exp/tts_train_teacher_raw_phn_none_cn_hubert \
    --vocoder_file ${vocoder_file} \
    ${opts} "$@"

# --inference_args "--use_teacher_forcing true" \


