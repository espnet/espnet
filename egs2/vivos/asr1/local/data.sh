#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./db.sh
. ./path.sh
. ./cmd.sh

datadir=${VIVOS}
vivos_root=${datadir}/vivos
data_url=https://zenodo.org/api/files/a3a96378-5e63-4bf3-8fa6-fe2bebc871c7/vivos.tar.gz

ndev_utt=100

train_vivos="train"
train_set="train_nodev"
train_dev="train_dev"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data download"

    mkdir -p ${datadir}

    if [ -d ${datadir}/vivos ] || [ -f ${datadir}/vivos.tar.gz ]; then
        log "$0: vivos directory or archive already exists in ${datadir}. Skipping download."
    else
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 1;
        fi
        log "$0: downloading data from ${data_url}"

        if ! wget --no-check-certificate ${data_url} -P ${datadir}; then
            log "$0: error executing wget ${data_url}"
            exit 1;
        fi

        if ! tar -xvzf ${datadir}/vivos.tar.gz -C ${datadir}; then
            log "$0: error un-tarring archive ${datadir}/vivos.tar.gz"
            exit 1;
        fi

        log "$0: Successfully downloaded and un-tarred ${datadir}/vivos.tar.gz"
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    mkdir -p data/{train,test}

    if [ ! -f ${vivos_root}/README ]; then
        log "Cannot find vivos root! Exiting..."
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
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Splitting train set into train+dev set"

    utils/subset_data_dir.sh --first            \
                             data/${train_vivos}\
                             ${ndev_utt}        \
                             data/${train_dev}

    train_utt=$(($(wc -l < data/${train_vivos}/text) - ndev_utt))

    utils/subset_data_dir.sh --last             \
                             data/${train_vivos}\
                             ${train_utt}       \
                             data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
