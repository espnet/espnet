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
train_dir=dump/raw_tts_librispeech/test_clean
valid_dir=dump/raw_tts_librispeech/test_clean
train_infer_dir=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/decode_inhouse_valid.total_count.ave_5best.till100epoch/tts_test_clean/
valid_infer_dir=exp/speechlm_ls_giga_mlsen_train_multiscale_1b/decode_inhouse_valid.total_count.ave_5best.till100epoch/tts_test_clean/
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
        src_dir=${part}_dir
        src_dir=${!src_dir}
        
        tgt_dir=${src_dir}_hfrl_${tag} 
        mkdir -p ${tgt_dir}
        
        infer_dir=${part}_infer_dir
        infer_dir=${!infer_dir}

        # TODO(Jinchuan): add sample selection strategy

        # Generate a new data.json file based on the original data.json file, so
        # that the new data.json file can be used for HFRL training.
        # (1) change the speaker prompt (utt2spk) as fixed codec tokens rather
        #     than the original utt-to-speaker mapping.
        # (2) add sampled.scp as an extra data file to specify the sampled items

        cp ${infer_dir}/utt2spk_token.scp ${tgt_dir}/utt2spk
        ./pyscripts/utils/convert_to_multicol_kaldi_ark.py \
          --input_scp ${infer_dir}/wav.scp_token.scp \
          --output_scp ${tgt_dir}/sampled.scp \
          --task ${task}

        ./pyscripts/utils/speechlm_add_data_entry.py \
          --input_json ${src_dir}/data.json \
          --output_json ${tgt_dir}/data.json \
          --path_name_types "${tgt_dir}/utt2spk,spk,kaldi_ark" \
          --path_name_types "${tgt_dir}/sampled.scp,codec,multicol_kaldi_ark"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Training ..."

    # Only take care of the training. The inference will have the same
    # format as usual.
    train_args+="--init_param ${resume}"
    ./speechlm.sh \
        --stage 7 --stop_stage 7 \
        --skip_data_prep true \
        --data_combo_name $(basename ${train_dir})_hfrl_${tag} \
        --token_list_dir ${token_list_dir} \
        --ngpu ${ngpu} \
        --nj ${nj} \
        --cleaner ${cleaner} \
        --g2p ${g2p} \
        --audio_format ${audio_format} \
        --train_config ${train_config} \
        --train_jsons ${train_dir}_hfrl_${tag}/data.json \
        --valid_jsons ${valid_dir}_hfrl_${tag}/data.json \
        --train_args "${train_args}" \
        "$@"
fi