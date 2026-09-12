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
Usage: $0 [--stage <stage>] [--stop_stage <stage>] [--force true|false]

Stages:
    1: Download and extract OpenSLR SLR45.
    2: Prepare Kaldi-style train/dev/test data directories.
EOF
)

SECONDS=0
stage=1
stop_stage=2
force=false
data_dir=data

log "$0 $*"
. ./path.sh
. ./cmd.sh
. ./db.sh

. utils/parse_options.sh

if [ -z "${ST_AEDS}" ]; then
    log "Error: ST_AEDS is not set."
    exit 2
fi

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

download_args=("${ST_AEDS}")
if "${force}"; then
    download_args+=(--force)
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "stage 1: Download and extract ST-AEDS to ${ST_AEDS}"
    ./local/download_and_extract.sh "${download_args[@]}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "stage 2: Prepare Kaldi-style data directories under ${data_dir}"
    python3 ./local/prepare_data.py --root "${ST_AEDS}" --data-dir "${data_dir}"
    for dset in train dev test; do
        utils/utt2spk_to_spk2utt.pl "${data_dir}/${dset}/utt2spk" > "${data_dir}/${dset}/spk2utt"
        utils/fix_data_dir.sh "${data_dir}/${dset}"
        utils/validate_data_dir.sh --no-feats "${data_dir}/${dset}"
    done
    python3 ./local/check_text_overlap.py --data-dir "${data_dir}" train dev test
fi

log "Successfully finished data preparation. [elapsed=${SECONDS}s]"
