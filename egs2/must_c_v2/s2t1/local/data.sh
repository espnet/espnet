#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

. ./db.sh || exit 1
. ./path.sh || exit 1
. ./cmd.sh || exit 1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000

datasets="dev train"
langs="de ja zh"
max_sec=30
resolution=0.02

log "$0 $*"
. utils/parse_options.sh

if [ -z "${MUST_C}" ]; then
    log "Fill the value of 'MUST_C' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${MUST_C}" ]; then
    log "stage 1: Data Download"
    mkdir -p ${MUST_C}
    for lang in ${langs}; do
        local/download_and_untar.sh ${MUST_C} ${lang} "v2"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python local/data_prep.py \
        --data_path ${MUST_C} \
        --output_path ./data \
        --datasets ${datasets} \
        --langs ${langs} \
        --max_sec ${max_sec} \
        --resolution ${resolution}

    for x in ${datasets}; do
        utils/fix_data_dir.sh --utt_extra_files "text.prev text.ctc" data/${x} || exit 1
        utils/validate_data_dir.sh --no-feats --non-print data/${x} || exit 1
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
