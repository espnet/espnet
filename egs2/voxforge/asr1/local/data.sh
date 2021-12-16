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
help_message=$(cat << EOF
Usage: $0

Options:
    --lang (str): it, de, en, es, fr, it, nl, pt, ru
EOF
)
SECONDS=0


stage=1
stop_stage=100000
lang=it # de, en, es, fr, it, nl, pt, ru


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${VOXFORGE}/${lang}/extracted" ]; then
        log "stage 1: Download data to ${VOXFORGE}"
        local/getdata.sh "${lang}" "${VOXFORGE}"
    else
        log "stage 1: ${VOXFORGE}/${lang}/extracted is already existing. Skip data downloading"
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    selected=${VOXFORGE}/${lang}/extracted
    # Initial normalization of the data
    local/voxforge_data_prep.sh --flac2wav false "${selected}" "${lang}"
    local/voxforge_format_data.sh "${lang}"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Split all_${lang} into data/tr_${lang} data/dt_${lang} data/et_${lang}"
    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_"${lang}" data/tr_"${lang}" data/dt_"${lang}" data/et_"${lang}"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
