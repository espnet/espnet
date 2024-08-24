#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=24000
fmin=0
fmax=22050
n_fft=2048
n_shift=300
win_length=1200

score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

opts="--audio_format wav "

train_set=tr_no_dev
valid_set=dev
test_sets="eval"

# training and inference configuration
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=None
cleaner=none

pitch_extract=dio
ying_extract=None

combine_path=""
combine_path+="$(realpath ../../itako/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../opencpop/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../acesinger/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../kising/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../m4singer/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../ameboshi/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../kiritan/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../itako/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../oniku_kurumi_utagoe_db/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../ofuton_p_utagoe_db/svs1/dump/raw/)"

use_sid=true
use_lid=true

min_wav_duration=2.0

./svs.sh \
    --lang zh_jp \
    --svs_task gan_svs \
    --local_data_opts "--combine_path ${combine_path} --stage 1" \
    --feats_type raw \
    --use_sid ${use_sid} \
    --use_lid ${use_lid} \
    --pitch_extract "${pitch_extract}" \
    --ying_extract "${ying_extract}" \
    --fs "${fs}" \
    --fmax "${fmax}" \
    --fmin "${fmin}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --min_wav_duration ${min_wav_duration} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    --write_collected_feats true \
    ${opts} "$@"
