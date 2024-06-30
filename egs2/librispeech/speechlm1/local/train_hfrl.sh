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

train_config=conf/train_multiscale_1b_dpo.yaml

# original dataset and sampled examples
train_set=dump/raw_tts_librispeech/train_clean_100
valid_set=dump/raw_tts_librispeech/dev_clean
train_samples=exp/speechlm_ls_giga_mlsen_train_multiscale/topk20_temp1.5/tts_train_clean_100/wav.scp
valid_samples=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/decode_inhouse_valid.total_count.ave_5best.till100epoch/tts_dev_clean/wav.scp
token_list_dir=data/token_list/ls_giga_mlsen

# Tokenization options:
fs=16000
audio_format=flac.ark
codec_choice="inhouse" # codec
codec_checkpoint_path=null
codec_config_path=null
codec_hf_model_tag=null

# HFRL options
tag=sample10
task="tts"
train_args=
resume=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/valid.total_count.ave_5best.till100epoch.pth

# Other options
nj=32
ngpu=8
g2p="g2p_en_no_space"
cleaner="tacotron"
stage=2
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

if [ -z ${tag} ]; then
    echo "tag is needed ... " && exit 1;
fi

if [ -z ${resume} ]; then
    echo "Resume checkpoint is needed ... " && exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Prepare dataset for training"
    for part in train valid; do
        dset=${part}_set
        sample_file=${part}_samples

        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "${!sample_file}" "${!dset}_hfrl_${tag}/audio_raw"
        
        scripts/feats/codec_tokenization.sh \
            --src_dir ${!dset}_hfrl_${tag}/audio_raw \
            --tgt_dir ${!dset}_hfrl_${tag} \
            --codec_fs ${fs} \
            --dump_audio false \
            --file_name  wav.scp \
            --nj ${nj} \
            --codec_choice ${codec_choice} \
            --checkpoint_path ${codec_checkpoint_path} \
            --config_path ${codec_config_path} \
            --hf_model_tag ${codec_hf_model_tag}
        cp ${!dset}_hfrl_${tag}/wav.scp ${!dset}_hfrl_${tag}/sampled.scp

        ./pyscripts/utils/speechlm_add_data_entry.py \
          --input_json ${!dset}/data.json \
          --output_json ${!dset}_hfrl_${tag}/data.json \
          --path_name_type "${!dset}_hfrl_${tag}/sampled.scp,sampled,kaldi_ark"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Training ..."

    # Only take care of the training. The inference will have the same
    # format as usual.
    train_args+="--init_param ${resume}"
    ./speechlm.sh \
        --stage 8 --stop_stage 8 \
        --skip_data_prep true \
        --data_combo_name $(basename ${train_set})_hfrl_${tag} \
        --token_list_dir ${token_list_dir} \
        --ngpu ${ngpu} \
        --nj ${nj} \
        --cleaner ${cleaner} \
        --g2p ${g2p} \
        --audio_format ${audio_format} \
        --train_config ${train_config} \
        --train_jsons ${train_set}_hfrl_${tag}/data.json \
        --valid_jsons ${valid_set}_hfrl_${tag}/data.json \
        --train_args ${train_args} \
        "$@"
fi