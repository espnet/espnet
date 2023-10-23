#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=2

. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../ru_open_stt/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in asr_calls_2_val buriy_audiobooks_2_val dev public_youtube700_val train; do
        utils/fix_data_dir.sh ../../ru_open_stt/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../ru_open_stt/asr1/data/${part} \
            --output_dir data/ru_open_stt/${part}_whisper \
            --prefix ru_open_stt \
            --src rus \
            --src_field 2 \
            --num_proc 100
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/ru_open_stt/${part}_whisper
    done
fi
