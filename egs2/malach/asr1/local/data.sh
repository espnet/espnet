#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=3

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${MALACH}" ]; then
    log "Fill the value of 'MALACH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for x in train dev; do
	mkdir -p data/${x}
	cp ${MALACH}/data/${x}/{segments,text,utt2spk} data/${x}/
	sed -e "s|/speech7/picheny5_nb/testi/927/kaldi/egs/malach/ldc|${MALACH}/data|" ${MALACH}/data/${x}/wav.scp > data/${x}/wav.scp
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # make a development set
    utils/subset_data_dir.sh --first data/train 4000 data/dev_4k
    n=$(($(wc -l < data/train/text) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
