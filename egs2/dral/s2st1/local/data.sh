#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

src_lang=$1 # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
tgt_lang=$2
RAW_DIR=raw
time_type=long
output_dir=$3
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "Download data from DRAL into $RAW_DIR"
python3 local/prepare_datasets.py download conf/dataset_urls.yaml $RAW_DIR
log "Extract $src_lang-$tgt_lang into $output_dir"
python3 local/prepare_datasets.py  extract_recordings $RAW_DIR $src_lang $tgt_lang $time_type $output_dir

log "Successfully finished. [elapsed=${SECONDS}s]"
