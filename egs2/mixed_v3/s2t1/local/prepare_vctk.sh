#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=2

. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../vctk/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in dev  eval1  tr_no_dev; do
        utils/fix_data_dir.sh ../../vctk/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../vctk/asr1/data/${part} \
            --output_dir data/vctk/${part}_whisper \
            --prefix VCTK \
            --src eng \
            --src_field 0 \
            --num_proc 100
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/vctk/${part}_whisper
    done
fi
