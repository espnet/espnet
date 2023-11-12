#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100

src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
tgt_lang=en
RAW_DIR=raw
time_type=long
output_dir=data
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${RAW_DIR}"
    log "Prepare source data from commonvoice 4.0"
    python3 local/prepare_datasets.py download conf/dataset_urls.yaml $RAW_DIR
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for commonvoice and cvss"
    python3 local/prepare_datasets.py  extract_recordings $RAW_DIR $src_lang $tgt_lang $time_type $output_dir
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
