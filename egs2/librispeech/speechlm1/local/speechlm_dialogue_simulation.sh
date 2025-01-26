#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Create spoken dialogue SFT dataset from the following dataset:

# SmolTalk:
# https://huggingface.co/datasets/HuggingFaceTB/smoltalk

# SODA:
# https://huggingface.co/datasets/allenai/soda

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=2
stop_stage=2
nj=40
tag=sft_acl

# Input and output folders / files
text_output_dir=dump/raw_text_dialogue_${tag}
audio_output_dir=dump/raw_audio_dialogue_${tag}
assistant_prompt_list=data/prompt_librispeech_test_clean_100.scp
user_prompt_list=dump/raw_codec_ssl_tts_yodas/train_yodas/index_files/wav.scp

# TTS setup
tag=full_tts_target
inference_model=11_16epoch.pth
expdir=exp_publish
inference_config=conf/decode_tts_espnet.yaml
nbest=5
tts_nj=1

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Generate text SFT data"

    python local/speechlm_dialogue_simulation.py \
      --input_hf_tag HuggingFaceTB/smoltalk \
      --output_dir ${text_output_dir} || exit 1;

    python local/speechlm_dialogue_simulation.py \
      --input_hf_tag allenai/soda \
      --output_dir ${text_output_dir} || exit 1;

    python local/speechlm_dialogue_simulation.py \
      --input_hf_tag HuggingFaceH4/ultrachat_200k \
      --output_dir ${text_output_dir} || exit 1;
    
    for subset in `ls ${text_output_dir}`; do
        cp ${text_output_dir}/${subset}/data/dialogue.1 ${text_output_dir}/${subset}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${text_output_dir}/${subset}/data.json \
          --file_modality_type "${text_output_dir}/${subset}/dialogue,dialogue,dialogue_json" || exit 1;
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Generate audio_dialogue SFT data."
    
    # SODA
    python local/speechlm_dialogue_simulation.py \
      --input_hf_tag allenai/soda \
      --task audio_dialogue \
      --output_dir ${audio_output_dir}_original \
      --assistant_prompt_list ${assistant_prompt_list} \
      --user_prompt_list ${user_prompt_list} || exit 1;
    
    for tts_dir in `find ${audio_output_dir}_original -name rank*pack* | grep tts_simulation`; do
        if [ ! -f ${tts_dir}/data.json ]; then
            echo "working on ${tts_dir}"
            python pyscripts/utils/make_speechlm_json.py \
              --task codec_ssl_tts \
              --output_json ${tts_dir}/data.json \
              --file_modality_type ${tts_dir}/text,text_bpe,text \
              --file_modality_type ${tts_dir}/utt2spk,spk,kaldi_ark \
              --file_modality_type ${tts_dir}/wav.scp,codec_ssl,kaldi_ark || exit 1;
        fi
    done
fi

