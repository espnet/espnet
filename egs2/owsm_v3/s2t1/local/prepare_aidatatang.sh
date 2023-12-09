#!/usr/bin/env bash
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
nproc=10
stage=1
stop_stage=2

log "$0 $*"

. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../aidatatang_200zh/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in train dev test; do
        utils/fix_data_dir.sh ../../aidatatang_200zh/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../aidatatang_200zh/asr1/data/${part} \
            --output_dir data/aidatatang/${part}_whisper \
            --prefix AIDATATANG_200ZH \
            --src zho \
            --src_field 0 \
            --num_proc ${nproc}
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
        data/aidatatang/${part}_whisper
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
