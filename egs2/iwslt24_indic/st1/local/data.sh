#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

tgt_lang=$1  # one of hi (Hindi), bn (Bengali), or ta (Tamil)
remove_archive=false

log "$0 $*"
. utils/parse_options.sh

if [ -z "${IWSLT24_INDIC}" ]; then
    log "Please fill the value of 'IWSLT24_INDIC' of db.sh to indicate where the dataset zip files are downloaded."
    exit 1
fi

# check if moses is installed
if ! command -v tokenizer.perl > /dev/null; then
    log "Error: The moses tool is not installed. Please install moses as follows: cd ${MAIN_ROOT}/tools && make moses.done"
    exit 1
fi

if [ $# -ne 1 ]; then
    log "Usage: $0 <tgt_lang>"
    log "e.g.: $0 hi"
    exit 1
fi

# check tgt_lang
tgt_langs="hi_bn_ta"
if ! echo "${tgt_langs}" | grep -q "${tgt_lang}"; then
    log "Error: ${tgt_lang} is not supported. It must be one of hi, bn, or ta."
    exit 1;
fi

log "Checking download and unpacking the dataset..."
mkdir -p ${IWSLT24_INDIC}
local/download_and_unpack.sh ${IWSLT24_INDIC} ${tgt_lang} ${remove_archive}

# ensure new line at the end of file
for split in train dev; do
    for ext in en ${tgt_lang} yaml; do
        filename=${IWSLT24_INDIC}/en-${tgt_lang}/data/${split}/txt/${split}.${ext}
        # shellcheck disable=SC1003
        sed -i -e '$a\' "${filename}"
    done
done
# shellcheck disable=SC1003
sed -i -e '$a\' "${IWSLT24_INDIC}/en-${tgt_lang}/data/tst-COMMON/txt/tst-COMMON.yaml"

log "Preparing data in ESPnet format..."
local/data_prep.sh ${IWSLT24_INDIC} ${tgt_lang}

log "Successfully finished data preparation. [elapsed=${SECONDS}s]"
