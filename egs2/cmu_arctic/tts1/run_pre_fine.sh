#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
set -evx
fs=16000
n_fft=1024
n_shift=256
win_length=null

inference_config=conf/decode.yaml

# g2p=g2p_en # Include word separator
g2p=g2p_en_no_space # Include no word separator


pretrain_stage=true
adapt_stage=true
# If you want to run only one of the stages (e.g., the adaptation stage),
# set "false" to the one you don't want to run (e.g., the pre-training stage)

if [[ ${pretrain_stage} == "true" ]]; then
    spk=all
    train_set=${spk}_train_no_dev
    valid_set=${spk}_dev
    test_sets=${spk}_eval
    train_config=conf/tuning/train_transformer_pre.yaml
    opts="--audio_format wav --local_data_opts ${spk} "

    ./tts.sh \
        --lang en \
        --feats_type raw \
        --fs "${fs}" \
        --n_fft "${n_fft}" \
        --n_shift "${n_shift}" \
        --win_length "${win_length}" \
        --token_type phn \
        --cleaner tacotron \
        --g2p "${g2p}" \
        --train_config "${train_config}" \
        --inference_config "${inference_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --srctexts "data/${train_set}/text" \
        ${opts} "$@"
fi

# Modify "--train_args "--init_param <path of the pre-trained model>""
# according to the actual path of your experiment.
if [[ ${adapt_stage} == "true" ]]; then
    spk=slt
    train_set=${spk}_train_no_dev
    valid_set=${spk}_dev
    test_sets=${spk}_eval
    train_config=conf/tuning/finetune_transformer.yaml
    opts="--audio_format wav --local_data_opts ${spk} "
    ./tts.sh \
        --lang en \
        --feats_type raw \
        --fs "${fs}" \
        --n_fft "${n_fft}" \
        --n_shift "${n_shift}" \
        --win_length "${win_length}" \
        --token_type phn \
        --cleaner tacotron \
        --g2p "${g2p}" \
        --train_config "${train_config}" \
        --train_args "--init_param exp/tts_train_transformer_pre_raw_phn_tacotron_g2p_en_no_space/train.loss.best.pth" \
        --inference_config "${inference_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --srctexts "data/${train_set}/text" \
        ${opts} "$@"
fi
