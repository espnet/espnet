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
data_url=http://www.openslr.org/resources/1/waves_yesno.tar.gz
ndev_utt=100

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_yesno="train_yesno"
train_set="train_nodev"
train_dev="train_dev"
eval_set="test_yesno"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage 1: Data Download"

    if [ ! -d waves_yesno ]; then
        wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;

        tar -xvzf waves_yesno.tar.gz || exit 1;
        rm ./waves_yesno/README*
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    mkdir -p data/local
    
    ls -1 waves_yesno > data/local/waves_all.list

    local/create_yesno_waves_test_train.pl data/local/waves_all.list    \
                                           data/local/waves.test        \
                                           data/local/waves.train

    local/create_yesno_wav_scp.pl waves_yesno   \
                                  data/local/waves.test > data/local/${eval_set}_wav.scp

    local/create_yesno_wav_scp.pl waves_yesno   \
                                  data/local/waves.train > data/local/train_yesno_wav.scp

    local/create_yesno_txt.pl data/local/waves.test > data/local/${eval_set}.txt

    local/create_yesno_txt.pl data/local/waves.train > data/local/train_yesno.txt


    for x in train_yesno test_yesno; do
        mkdir -p data/$x

        cp data/local/${x}_wav.scp data/$x/wav.scp
        cp data/local/$x.txt data/$x/text

        cat data/$x/text | awk '{printf("%s global\n", $1);}' > data/$x/utt2spk
        utils/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    dev_utt=2
    
    utils/subset_data_dir.sh --first            \
                             data/train_yesno   \
                             ${dev_utt}         \
                             data/${train_dev}

    train_utt=$(($(wc -l < data/train_yesno/text) - 2))
    
    utils/subset_data_dir.sh --last             \
                             data/train_yesno   \
                             ${train_utt}       \
                             data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
