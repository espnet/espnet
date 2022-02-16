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

datadir=${YESNO}
yesno_root=${datadir}/waves_yesno
data_url=http://www.openslr.org/resources/1/waves_yesno.tar.gz

ndev_utt=2

train_set="train_nodev"
train_dev="train_dev"
eval_set="test_yesno"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data download"

    mkdir -p ${datadir}

    if [ -d ${datadir}/waves_yesno ] || [ -f ${datadir}/waves_yesno.tar.gz ]; then
        log "$0: yesno directory or archive already exists in ${datadir}. Skipping download."
    else
        if ! command -v wget >/dev/null; then
            log "$0: wget is not installed."
            exit 1;
        fi
        log "$0: downloading data from ${data_url}"

        if ! wget --no-check-certificate ${data_url} -P "${datadir}"; then
            log "$0: error executing wget ${data_url}"
            exit 1;
        fi

        if ! tar -xvzf ${datadir}/waves_yesno.tar.gz -C "${datadir}"; then
            log "$0: error un-tarring archive ${datadir}/waves_yesno.tar.gz"
            exit 1;
        fi
        rm ${yesno_root}/README*

        log "$0: Successfully downloaded and un-tarred ${datadir}/waves_yesno.tar.gz"
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    mkdir -p data/local
    ls -1 ${yesno_root} > data/local/waves_all.list

    local/create_yesno_waves_test_train.pl data/local/waves_all.list    \
                                           data/local/waves.test        \
                                           data/local/waves.train

    local/create_yesno_wav_scp.pl ${yesno_root} \
                                  data/local/waves.test > data/local/${eval_set}_wav.scp

    local/create_yesno_wav_scp.pl ${yesno_root} \
                                  data/local/waves.train > data/local/train_yesno_wav.scp

    local/create_yesno_txt.pl data/local/waves.test > data/local/${eval_set}.txt

    local/create_yesno_txt.pl data/local/waves.train > data/local/train_yesno.txt

    for x in train_yesno test_yesno; do
        mkdir -p data/$x

        cp data/local/${x}_wav.scp data/$x/wav.scp
        cp data/local/$x.txt data/$x/text

        <data/$x/text awk '{printf("%s global\n", $1);}' > data/$x/utt2spk
        utils/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Splitting train set into train+dev set"

    utils/subset_data_dir.sh --first            \
                             data/train_yesno   \
                             ${ndev_utt}        \
                             data/${train_dev}

    train_utt=$(($(wc -l < data/train_yesno/text) - ndev_utt))

    utils/subset_data_dir.sh --last             \
                             data/train_yesno   \
                             ${train_utt}       \
                             data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
