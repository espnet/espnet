#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Prepare text LM training corpus Pipe-Uncopyrighted into Espnet format
# Dataset website: https://huggingface.co/datasets/monology/pile-uncopyrighted

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


stage=2
stop_stage=3
data_tag=monology/pile-uncopyrighted
tokenizer_tag=EleutherAI/pythia-1b
portion=0.2

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Download data ... "
    huggingface-cli download \
      --repo-type dataset \
      --local-dir ./data/local/pile_uncopyrighted \
      ${data_tag}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Process downloaded files"
    all_files=`find ./data/local/pile_uncopyrighted -name *.jsonl.zst`
    for file in ${all_files}; do
        file_name=$(basename ${file})
        clean_name=${file_name%.jsonl.zst}
        zstdcat ${file} | jq -c '.text' | sed 's/^"\(.*\)"$/\1/' |\
            awk -v prefix="$clean_name" -v p="${portion}" \
            'BEGIN {srand()} {if (NF <= 1000 && rand() <= p) {print prefix "_" NR " " $0}}' \
            > $(dirname ${file})/${clean_name} &
    done; wait
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Aggregate Results ... "
    mkdir -p data/{train,val,test}
    cp ./data/local/pile_uncopyrighted/val ./data/val/text
    cp ./data/local/pile_uncopyrighted/test ./data/test/text
    rm -f ./data/train/text
    for file in `ls ./data/local/pile_uncopyrighted/train | grep -v .jsonl.zst`; do 
        cat ./data/local/pile_uncopyrighted/train/${file} >> ./data/train/text
    done
fi