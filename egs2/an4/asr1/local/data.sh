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

stage=1
stop_stage=100

datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/
data_url2=https://huggingface.co/datasets/espnet/an4/resolve/main
ndev_utt=100

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train_nodev"
train_dev="train_dev"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    mkdir -p ${datadir}
    if ! local/download_and_untar.sh ${datadir} ${data_url}; then
        log "Failed to download from the original site, try a backup site."
        local/download_and_untar.sh ${datadir} ${data_url2}
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python3 local/data_prep.py ${an4_root} sph2pipe

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
    done

    # make a dev set
    utils/subset_data_dir.sh --first data/train "${ndev_utt}" "data/${train_dev}"
    n=$(($(wc -l < data/train/text) - ndev_utt))
    utils/subset_data_dir.sh --last data/train "${n}" "data/${train_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
