#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=2
stop_stage=2
nproc=10

. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../wsj/asr1/
    ./local/data.sh
    # Override the text file: keep the original text but exclude the special tokens
    ./local/wsj_format_data.sh --whisper_text_norm true
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data/wsj
    nlsyms_file=data/wsj/nlsyms.txt
    cut -f 2- ../../wsj/asr1/data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms_file}
    nlsyms=$(cat ${nlsyms_file} | tr '\n' ' ')

    utt_extra_files="text.prev text.ctc"
    for part in train_si284 test_dev93 test_eval92; do
        utils/fix_data_dir.sh ../../wsj/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../wsj/asr1/data/${part} \
            --output_dir data/wsj/${part}_whisper \
            --prefix WSJ \
            --src eng \
            --src_field 3 \
            --num_proc ${nproc} \
            --nlsyms ${nlsyms}
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" \
        data/wsj/${part}_whisper
    done
fi
