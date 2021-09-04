#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

spk=jvs010
train_set=${spk}_tr_no_dev
valid_set=${spk}_dev
test_sets="${spk}_dev ${spk}_eval1"

train_config=conf/finetune.yaml
inference_config=conf/decode.yaml

# Input example: こ、こんにちは

# 1. Phoneme + Pause
# (e.g. k o pau k o N n i ch i w a)
g2p=pyopenjtalk

# 2. Kana + Symbol
# (e.g. コ 、 コ ン ニ チ ワ)
# g2p=pyopenjtalk_kana

# 3. Phoneme + Accent
# (e.g. k 1 0 o 1 0 k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
# g2p=pyopenjtalk_accent

# 4. Phoneme + Accent + Pause
# (e.g. k 1 0 o 1 0 pau k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
# g2p=pyopenjtalk_accent_with_pause

# 5. Phoneme + Prosody symbols
# (e.g. ^, k, #, o, _, k, o, [, N, n, i, ch, i, w, a, $)
# g2p=pyopenjtalk_prosody

./tts.sh \
    --lang jp \
    --local_data_opts "--spk ${spk}" \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner jaconv \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
