#!/usr/bin/env bash

# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../fisher_callhome_spanish/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in train dev fisher_dev fisher_dev2 fisher_test callhome_evltest callhome_devtest; do
        utils/fix_data_dir.sh ../../fisher_callhome_spanish/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../fisher_callhome_spanish/asr1/data/${part} \
            --output_dir data/fisher_callhome/${part}_whisper \
            --prefix FISHER_CALLHOME_SPANISH \
            --src spa \
            --src_field 6 \
            --num_proc 10
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/fisher_callhome/${part}_whisper
    done
fi
