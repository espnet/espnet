#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Modified by flylili001

set -euo pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=-1       # start from -1 if you need to start from data download
stop_stage=100

. utils/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    local/download_data.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases

    local/prepare_data.sh

fi

