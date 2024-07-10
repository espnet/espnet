#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Based on existing SpeechLM dataset, add additional data items to support training
# such as HFRL (DPO, PPO etc.)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# original dataset and sampled examples
train_dir=dump/raw_tts_librispeech/train_960
valid_dir=dump/raw_tts_librispeech/dev_clean
train_sampling_dir=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/decode_inhouse_valid.total_count.ave_5best.till100epoch/tts_train_960
valid_sampling_dir=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/decode_inhouse_valid.total_count.ave_5best.till100epoch/tts_dev_clean
token_list_dir=data/token_list/ls_giga_mlsen

# Tokenization options:
fs=16000
audio_format=flac.ark
codec_choice="inhouse" # codec
codec_checkpoint_path=null
codec_config_path=null
codec_hf_model_tag=null

# Common Config for SFT and HFRL
tag=
data_combo_name=
task="tts"
pretrain_checkpoint=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/valid.total_count.ave_5best.till100epoch.pth
select_metrics="spk_similarity"

# SFT options
use_sft=true
sft_train_args=
sft_config=conf/train_multiscale_1b_sft.yaml

# HFRL options
hfrl_train_args=
hfrl_config=conf/train_multiscale_1b_dpo.yaml
use_reflm=true

# Other options
nj=32
ngpu=8
g2p="g2p_en_no_space"
cleaner="tacotron"
stage=1
stop_stage=100

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z ${pretrain_checkpoint} ]; then
    echo "Pretrained checkpoint is needed ... " && exit 1;
fi

if [ -z ${tag} ]; then
    tag=metric_$(echo "${select_metrics}" | tr ' ' '_')
fi

if [ -z ${data_combo_name} ]; then
    data_combo_name=metric_$(echo "${select_metrics}" | tr ' ' '_')
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Prepare dataset for training"
    for part in valid train; do
        src_dir=${part}_dir
        src_dir=${!src_dir}
        
        tgt_dir=${src_dir}_hfrl_${data_combo_name}
        mkdir -p ${tgt_dir}
        
        sampling_dir=${part}_sampling_dir
        sampling_dir=${!sampling_dir}

        metric_opts=
        for metric in ${select_metrics}; do
            if [ "${metric}" == "utmos" ]; then
                metric_opts+="--metric_names utmos "
                metric_opts+="--metric_weights 1.0 "
                metric_opts+="--metric_files ${sampling_dir}/eval_signal/utt_result.txt "
            elif [ "${metric}" == "spk_similarity" ]; then
                metric_opts+="--metric_names spk_similarity "
                metric_opts+="--metric_weights 1.0 "
                metric_opts+="--metric_files ${sampling_dir}/eval_spk/utt_result.txt "
            elif [ "${metric}" == "edit_distance" ]; then
                metric_opts+="--metric_names edit_distance "
                metric_opts+="--metric_weights 1.0 "
                metric_opts+="--metric_files ${sampling_dir}/eval_speech_wer/utt_result.txt "
            fi
        done

        if [ "${task}" == "tts" ]; then
            ./pyscripts/utils/build_hfrl_dataset.py \
            --ref_json ${src_dir}/data.json \
            --output_dir ${tgt_dir} \
            --path_modality_types "${src_dir}/text,g2p,text" \
            --path_modality_types "${sampling_dir}/utt2spk_token.scp,spk,kaldi_ark" \
            --path_modality_types "${sampling_dir}/wav.scp_token.scp,codec,multicol_kaldi_ark" \
            ${metric_opts}
        fi
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Training ..."

    if ${use_sft}; then
        # the checkpoint is used to initialize both corelm and reflm
        train_args="--init_param ${pretrain_checkpoint}:corelm:corelm"
        ./speechlm.sh \
            --stage 8 --stop_stage 8 \
            --tag sft_${tag} \
            --skip_data_prep true \
            --data_combo_name $(basename ${train_dir})_hfrl_${data_combo_name} \
            --token_list_dir ${token_list_dir} \
            --ngpu ${ngpu} \
            --nj ${nj} \
            --cleaner ${cleaner} \
            --g2p ${g2p} \
            --audio_format ${audio_format} \
            --train_config ${sft_config} \
            --train_jsons ${train_dir}_hfrl_${data_combo_name}/data.json \
            --valid_jsons ${valid_dir}_hfrl_${data_combo_name}/data.json \
            --train_args "${train_args}" \
            "$@"
    else
        echo "Skip SFT stage"
    fi
fi

if ${use_sft}; then
    pretrain_checkpoint=exp/speechlm_sft_${tag}/latest.pth
    tag=${tag}_with_sft
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Preference Alignment Training ..."

    # the checkpoint is used to initialize both corelm and reflm
    if ${use_reflm}; then
        train_args+="--init_param ${pretrain_checkpoint}:corelm:corelm ${pretrain_checkpoint}:corelm:reflm"
    else
        train_args+="--init_param ${pretrain_checkpoint}:corelm:corelm"
    fi
    ./speechlm.sh \
        --stage 8 --stop_stage 8 \
        --tag hfrl_${tag} \
        --skip_data_prep true \
        --data_combo_name $(basename ${train_dir})_hfrl_${data_combo_name} \
        --token_list_dir ${token_list_dir} \
        --ngpu ${ngpu} \
        --nj ${nj} \
        --cleaner ${cleaner} \
        --g2p ${g2p} \
        --audio_format ${audio_format} \
        --train_config ${hfrl_config} \
        --train_jsons ${train_dir}_hfrl_${data_combo_name}/data.json \
        --valid_jsons ${valid_dir}_hfrl_${data_combo_name}/data.json \
        --train_args "${train_args}" \
        "$@"
fi