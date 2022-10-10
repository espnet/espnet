#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
n_fft=1024
n_shift=256

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

# Data prep related
train_type=en_us # Use "en_us" or "multilingual".
token_type=phn

suffix=""
# Training the model only on the en_us single-speaker data
if [ "${train_type}" = en_us ]; then
    lang=en
    cleaner=tacotron
    g2p=g2p_en
    use_lid=false
    use_sid=false
    train_config=conf/train.yaml
elif [ "${train_type}" = multilingual ]; then
    # Training the model on the whole dataset
    lang=noinfo
    cleaner=none
    g2p=none
    use_lid=true
    use_sid=true
    if [ "${token_type}" = phn ]; then
        suffix="_phn"
    fi
    train_config=conf/train_multilingual.yaml
else
    log train_type: "${train_type} is not supported."
    exit 1
fi

local_data_opts+=" --train_type ${train_type}"

train_set=tr_no_dev${suffix}
valid_set=dev${suffix}
test_sets="dev${suffix} eval${suffix}"

inference_config=conf/decode.yaml

./tts.sh \
    --lang ${lang} \
    --local_data_opts "${local_data_opts}" \
    --feats_type raw \
    --use_lid ${use_lid} \
    --use_sid ${use_sid} \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type "${token_type}" \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_model valid.loss.best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
