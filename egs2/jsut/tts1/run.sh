#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
dev_set=dev
eval_set=eval1

tts_config=conf/train_tacotron2.v3.yaml
decode_config=conf/decode.yaml
# pyopenjtalk case: m i z u o m a r e e sh i a k a r a k a w a n a k U t e w a n a r a n a i n o d e s U
# pyopenjtalk_kana case: ミズヲマレーシアカラカワナクテワナラナイノデス。
g2p=pyopenjtalk
# g2p=pyopenjtalk_kana

# toke_type=char doesn't indicate kana, but mean kanji-kana-majiri-moji characters

./tts.sh \
    --feats_type fbank \
    --fs "${fs}" \
    --token_type phn \
    --cleaner jaconv \
    --g2p "${g2p}" \
    --train_config "${tts_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_set}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
