#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../swbd/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in train_nodup train_fisher train_dev eval2000; do
        utils/fix_data_dir.sh ../../swbd/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../swbd/asr1/data/${part} \
            --output_dir data/swbd/${part}_whisper \
            --prefix SWBD \
            --src eng \
            --src_field 7 \
            --num_proc 10
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/swbd/${part}_whisper
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
