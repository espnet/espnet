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

train_set="train_nodev"
train_dev="train_dev"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
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
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done

    # make a dev set
    utils/subset_data_dir.sh --first data/train 1 data/${train_dev}
    n=$(($(wc -l < data/train/text) - 1))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}

    # Create "test_seg" in order to test the use case of segments
    rm -rf data/test_seg
    utils/copy_data_dir.sh data/test data/test_seg
    <data/test/wav.scp awk '{ for(i=2;i<=NF;i++){a=a " " $i}; print($1 "_org", a) }' > data/test_seg/wav.scp
    cat << EOF > data/test_seg/segments
fcaw-cen8-b fcaw-cen8-b_org 0.0 2.9
mmxg-cen8-b mmxg-cen8-b_org 0.0 2.3
EOF

    # for enh task
    for x in test ${train_set} ${train_dev} test test_seg; do
        for f in wav.scp text utt2spk segments; do
            [ ! -f data/${x}/${f} ] && continue
            mv data/${x}/${f} data/${x}/${f}.old
            if [ $f = "segments" ]; then
                <data/${x}/${f}.old awk '{$1=$1"_SIMU"; $2=$2"_SIMU"; print($0)}' > data/${x}/${f}
            else
                <data/${x}/${f}.old awk '{$1=$1"_SIMU"; print($0)}' > data/${x}/${f}
            fi
            rm data/${x}/${f}.old
        done
        ln -s text data/${x}/text_spk1
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt

        cp data/${x}/wav.scp data/${x}/spk1.scp
        <data/${x}/wav.scp awk '{print($1, "SIMU")}' > data/${x}/utt2category
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
