#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000

lang=$1

log "$0 $*"
. utils/parse_options.sh

if [ -z "${MUST_C}" ]; then
    log "Fill the value of 'MUST_C' of db.sh"
    exit 1
fi

# check extra module installation
if ! command -v tokenizer.perl > /dev/null; then
    echo "Error: it seems that moses is not installed." >&2
    echo "Error: please install moses as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
    return 1
fi

if [ $# -ne 1 ]; then
    log "Error: ang argument is required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${MUST_C}" ]; then
    log "stage 1: Data Download"
    mkdir ${MUST_C}
    local/download_and_untar.sh ${MUST_C} ${lang} "v1"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    local/data_prep.sh ${MUST_C} ${lang} "v1"
    for set in train dev tst-COMMON tst-HE; do
        cp data/${set}.en-${lang}/text.lc.rm.en data/${set}.en-${lang}/text
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
