#!/usr/bin/env bash
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

stage=0
stop_stage=0

. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    python3 local/data_prep.py
fi
