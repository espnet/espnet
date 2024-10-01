#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Prepare ESPnet Text corpus for SpeechLM training.

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
stage=1
stop_stage=1
data_dir=/mnt/home/jinchuat/data/multilingual_cc_news
output_dir=dump/raw_textlm_cc_whole
tokenizer_tag=HuggingFaceTB/SmolLM-1.7B
max_len=8000
nj=6

log "$0 $*"
. utils/parse_options.sh

# . ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for file_path in `find ${data_dir} -name *.jsonl`; do
        file_name=$(basename ${file_path})
        file_name=${file_name%.jsonl}

        log "Processing ${file_path}"
        python3 pyscripts/text/huggingface_tokenize_text.py \
            --input_path ${file_path} \
            --output_dir ${output_dir}/${file_name} \
            --tokenizer_tag ${tokenizer_tag} \
            --nj ${nj} \
            --max_len ${max_len} &
    done; wait
fi
