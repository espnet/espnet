#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Create SFT dataset from smoltalk:
# https://huggingface.co/datasets/HuggingFaceTB/smoltalk

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

stage=0
stop_stage=100
text_output_dir=dump/raw_text_dialogue_smoltalk
audio_output_dir=dump

. ./db.sh
. ./path.sh

if [ -z "${SMOLTALK}" ]; then
    log "Fill the value of 'SMOLTALK' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Download smoltalk dataset ..."
    if [ -f ${SMOLTALK}/README.md ]; then
        log "Already have the smoltalk dataset. Skip downloading. "
    else
        huggingface-cli download --repo-type dataset --local-dir ${SMOLTALK} HuggingFaceTB/smoltalk
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Convert to ESPnet Dialogue data format"
    python local/sft_textlm_smoltalk.py \
      --input_dir ${SMOLTALK} \
      --output_dir ${text_output_dir} 
fi