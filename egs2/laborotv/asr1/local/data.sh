#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

#if [ ! -e "${LABOROTV}" ]; then
#    log "Fill the value of 'LABOROTV' of db.sh"
#    exit 1
#fi

TEDXJP_DATA_ROOT="tedx-jp"
train_set=train
train_dev=dev

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for x in `awk -F',' 'NR > 1 {print $3}' local/tedx-jp/tedx-jp-10k.csv`; do
	until youtube-dl \
	    --extract-audio \
	    --audio-format wav \
	    --write-sub \
	    --sub-format vtt \
	    --sub-lang ja \
	    --output "${TEDXJP_DATA_ROOT}/%(id)s.%(ext)s" \
	    ${x}
	do
	    # for some reason youtube-dl is quite unstable and we need to add
	    # the following resume process
	    echo "Try again"
	done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    local/tedx-jp/10k_data_prep.sh ${TEDXJP_DATA_ROOT}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Initial normalization of the data
    local/laborotv_data_prep.sh ${LABOROTV}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # make a development set during training by extracting the first 4000 utterances
    # followed by the CSJ recipe
    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # XXXhr XXmin
    n=$(($(wc -l < data/train/text) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set} # XXXh XXXmin
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
