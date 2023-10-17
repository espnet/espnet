#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

stage=1
stop_stage=2

. utils/parse_options.sh || exit 1;

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../magicdata/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in test dev train; do
        utils/fix_data_dir.sh ../../magicdata/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../magicdata/asr1/data/${part} \
            --output_dir data/magicdata/${part}_whisper \
            --prefix MagicData \
            --src zho \
            --src_field 0 \
            --num_proc 100
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/magicdata/${part}_whisper
    done
fi
