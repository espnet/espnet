#!/usr/bin/env bash
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
nproc=10
stage=0
stop_stage=2
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    pwd=${PWD}
    cd ../../ami/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utt_extra_files="text.prev text.ctc"
    for part in ihm_train ihm_dev ihm_eval; do
        utils/fix_data_dir.sh ../../ami/asr1/data/${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../ami/asr1/data/${part} \
            --output_dir data/ami/${part}_whisper \
            --prefix AMI \
            --src eng \
            --src_field 7 \
            --num_proc ${nproc} \
            --lower_case
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
          data/ami/${part}_whisper
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
