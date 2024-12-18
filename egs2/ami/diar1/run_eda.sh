#!/usr/bin/env bash

# Copyright 2021 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

train_config1="conf/train_diar_eda.yaml"
train_config2="conf/train_diar_eda_adapt.yaml"
decode_config_spk3="conf/decode_diar_eda_spk3.yaml"
decode_config_spk4="conf/decode_diar_eda_spk4.yaml"
decode_config_spk5="conf/decode_diar_eda_spk5.yaml"

pretrain_stage=true
adapt_stage=true
inference_stage=true

# options for /local/data/sh
setup_dir=ami_diarization_setup
mic_type=ihm
if_mini=false
sound_type=only_words
duration=20
min_wav_duration=0.0 # set to 0.0 to use all data, don't filter out short utterances

dumpdir=dump_eda
expdir=exp_eda

if [[ ${pretrain_stage} == "true" ]]; then
    # train diarization model with 4 speakers
    ./diar.sh \
        --collar 0.0 \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --ngpu 1 \
        --diar_config "${train_config1}" \
        --inference_config "${decode_config_spk4}" \
        --inference_nj 5 \
        --local_data_opts "\
            --stage 3 \
            --setup_dir ${setup_dir} \
            --num_spk 4 \
            --duration ${duration} \
            --mic_type ${mic_type} \
            --if_mini ${if_mini} \
            --sound_type ${sound_type} \
        " \
        --num_spk "4" \
        --min_wav_duration "${min_wav_duration}"\
        --dumpdir "${dumpdir}/spk4" \
        --expdir "${expdir}/spk4" \
        --diar_tag "train_diar_eda_raw_spk4"\
        --stop_stage 5 \
        "$@"
fi

# Modify "--diar_args "--init_param <path of the pre-trained model>""
# according to the actual path of your experiment.
if [[ ${adapt_stage} == "true" ]]; then
    ./diar.sh \
        --collar 0.0 \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --ngpu 1 \
        --diar_config "${train_config2}" \
        --inference_config "${decode_config_spk3}" \
        --inference_nj 5 \
        --local_data_opts "\
            --stage 3 \
            --setup_dir ${setup_dir} \
            --num_spk 3 \
            --duration ${duration} \
            --mic_type ${mic_type} \
            --if_mini ${if_mini} \
            --sound_type ${sound_type} \
        " \
        --diar_args "--init_param exp_eda/spk4/diar_train_diar_eda_raw_spk4/valid.acc.best.pth" \
        --diar_tag "train_diar_eda_adapt_raw_spk3" \
        --num_spk "3"\
        --dumpdir "${dumpdir}/spk3" \
        --expdir "${expdir}/spk3" \
        --stop_stage 5 \
        "$@"

    ./diar.sh \
        --collar 0.0 \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --ngpu 1 \
        --diar_config "${train_config2}" \
        --inference_config "${decode_config_spk5}" \
        --inference_nj 5 \
        --local_data_opts "\
            --stage 3 \
            --setup_dir ${setup_dir} \
            --num_spk 5 \
            --duration ${duration} \
            --mic_type ${mic_type} \
            --if_mini ${if_mini} \
            --sound_type ${sound_type} \
        " \
        --diar_args "--init_param exp_eda/spk3/diar_train_diar_eda_adapt_raw_spk3/latest.pth" \
        --diar_tag "train_diar_eda_adapt_raw_spk5" \
        --num_spk "5"\
        --dumpdir "${dumpdir}/spk5" \
        --expdir "${expdir}/spk5" \
        --stop_stage 5 \
        "$@"
fi

# Inferance, scoring, packing, and uploading Huggingface, use 4 spks data.
if [[ ${inference_stage} == "true" ]]; then
    ./diar.sh \
        --collar 0.0 \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --ngpu 1 \
        --diar_config "${train_config1}" \
        --inference_config "${decode_config_spk4}" \
        --inference_nj 5 \
        --local_data_opts "\
            --stage 3 \
            --setup_dir ${setup_dir} \
            --num_spk 4 \
            --duration ${duration} \
            --mic_type ${mic_type} \
            --if_mini ${if_mini} \
            --sound_type ${sound_type} \
        " \
        --num_spk "4" \
        --min_wav_duration "${min_wav_duration}"\
        --dumpdir "${dumpdir}/spk4" \
        --expdir "${expdir}/spk4" \
        --diar_args "--init_param exp_eda/spk5/diar_train_diar_eda_adapt_raw_spk5/latest.pth" \
        --diar_tag "train_diar_eda_raw_spk4"\
        --stage 6 \
        --stop_stage 9 \
        "$@"
fi
