#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
# . ./db.sh || exit 1;


train_set="train"
dev_set="val"
test_sets="test"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Data preparation via HuggingFace Datasets"
  python3 local/data_prep.py \
    --train_set "${train_set}" \
    --dev_set   "${dev_set}"   \
    --test_set  "${test_sets}" \
    --hf_repo   "MushanW/GLOBE_V2" \
    --dest_path "data"
fi

utils/fix_data_dir.sh data/${train_set}
utils/fix_data_dir.sh data/${dev_set}
utils/fix_data_dir.sh data/${test_sets}