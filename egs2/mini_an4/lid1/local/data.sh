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

an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train_minian4_nodev"
train_dev="train_minian4_dev"
sets="train_minian4_nodev train_minian4_dev"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train_minian4,test_minian4}

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python local/data_prep_lid.py ${an4_root} sph2pipe

    for x in test_minian4 train_minian4; do
        for f in wav.scp utt2lang; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2lang > data/${x}/lang2utt
    done

    # make a dev set
    mv data/train_minian4/utt2lang data/train_minian4/utt2spk
    utils/subset_data_dir.sh --first data/train_minian4 2 data/${train_dev}
    mv data/train_minian4_dev/utt2spk data/train_minian4_dev/utt2lang
    # here - 1 rather than - 2 is to ensure dev's langs in train
    n=$(($(wc -l < data/train_minian4/utt2spk) - 1))
    utils/subset_data_dir.sh --last data/train_minian4 ${n} data/${train_set}
    mv data/train_minian4_nodev/utt2spk data/train_minian4_nodev/utt2lang
    mv data/train_minian4/utt2spk data/train_minian4/utt2lang

    for set in ${sets}; do
        mv data/${set}/spk2utt data/${set}/lang2utt
    done

    find downloads/noise/ -iname "*.wav" | awk '{print "noise" NR " " $1}' > data/musan_music.scp
    find downloads/noise/ -iname "*.wav" | awk '{print "noise" NR " " $1}' > data/musan_noise.scp
    find downloads/noise/ -iname "*.wav" | awk '{print "noise" NR " " $1}' > data/musan_speech.scp
    find downloads/rirs/ -iname "*.wav" | awk '{print "rir" NR " " $1}' > data/rirs.scp
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
