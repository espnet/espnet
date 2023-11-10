#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./db.sh
. ./cmd.sh

stage=1
stop_stage=2
nproc=64

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../ksponspeech/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in dev eval_clean eval_other train; do
        echo "processing ${part}"
        utils/fix_data_dir.sh ../../ksponspeech/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../ksponspeech/asr1/data/${part} \
            --output_dir data/ksponspeech/${part}_whisper \
            --prefix KsponSpeech \
            --src kor \
            --src_field -1 \
            --num_proc ${nproc}
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/ksponspeech/${part}_whisper
    done
fi
