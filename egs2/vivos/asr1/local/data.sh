#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0

stage=-1
stop_stage=1

datadir=./downloads
vivos_root=${datadir}/vivos
data_url=https://ailab.hcmus.edu.vn/assets/vivos.tar.gz

ndev_utt=100

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_vivos="train"
train_set="train_nodev"
train_dev="train_dev"
eval_set="test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"

    mkdir -p ${datadir}

    if [ -d ${datadir}/vivos ] || [ -f ${datadir}/vivos.tar.gz ]; then
        echo "$0: vivos directory or archive already exists in ${datadir}. Skipping download."
    else
        if ! which wget >/dev/null; then
            echo "$0: wget is not installed."
            exit 1;
        fi
        echo "$0: downloading data from ${data_url}"

        cd ${datadir}
        if ! wget --no-check-certificate ${data_url}; then
            echo "$0: error executing wget ${data_url}"
            exit 1;
        fi

        if ! tar -xvzf vivos.tar.gz; then
            echo "$0: error un-tarring archive ${datadir}/vivos.tar.gz"
            exit 1;
        fi
        cd ..

        echo "$0: Successfully downloaded and un-tarred ${datadir}/vivos.tar.gz"
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p data/{train,test}

    if [ ! -f ${vivos_root}/README ]; then
        echo "Cannot find vivos root! Exiting..."
        exit 1;
    fi

    for x in test train; do
        awk -v dir=${vivos_root}/$x \
            '{ split($1,args,"_"); spk=args[1]; print $1" "dir"/waves/"spk"/"$1".wav" }' \
            ${vivos_root}/$x/prompts.txt | sort > data/$x/wav.scp

        awk '{ split($1,args,"_"); spk=args[1]; print $1" "spk }' \
            ${vivos_root}/$x/prompts.txt | sort > data/$x/utt2spk

        sort ${vivos_root}/$x/prompts.txt > data/$x/text
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    done

    echo "<blank>" > data/nlsyms.txt
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Splitting train set into train+dev set"

    utils/subset_data_dir.sh --first            \
                             data/${train_vivos}\
                             ${ndev_utt}        \
                             data/${train_dev}

    train_utt=$(($(wc -l < data/${train_vivos}/text) - ${ndev_utt}))

    utils/subset_data_dir.sh --last             \
                             data/${train_vivos}\
                             ${train_utt}       \
                             data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
