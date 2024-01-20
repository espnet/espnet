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
stop_stage=100000
data_url=www.openslr.org/resources/12
train_set="train"
train_dev="dev"

log "$0 $*"
. ./utils/parse_options.sh || exit 1;

. ./db.sh
. ./path.sh
. ./cmd.sh


ls100_datadir="data/librispeech_100"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Prepare LibriSpeech_100"
    . local/data_ls100.sh --stage 1 --stop-stage 3 || exit 1

    mkdir -p ${ls100_datadir}
    for dset in "train_clean_100" "dev"; do
        [ -d ${ls100_datadir}/${dset} ] && rm -r ${ls100_datadir:?}/${dset};
        mv data/${dset} ${ls100_datadir}
    done
fi

mlsuperb_datadir="data/ml_superb"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Prepare ML_SUPERB"
    . local/data_mlsuperb.sh --duration "1h" || exit 1

    mkdir -p ${mlsuperb_datadir}
    for dset in "train_1h" "dev_1h"; do
        [ -d ${mlsuperb_datadir}/${dset} ] && rm -r ${mlsuperb_datadir:?}/${dset};
        mv data/${dset} ${mlsuperb_datadir}
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Combine the training and valid sets"
    mkdir -p data/${train_set}
    mkdir -p data/${train_dev}
    rm data/${train_set}/* data/${train_dev}/*

    # librispeech_100
    prefix="librispeech-"
    for dset in "train_clean_100" "dev"; do
        if [ ${dset} = "train_clean_100" ]; then
            _dir="data/${train_set}"
        else
            _dir="data/${train_dev}"
        fi

        src_dir="data/librispeech_100/${dset}"
        <${src_dir}/utt2spk awk -v prefix="${prefix}" '{print(prefix $1, prefix $2)}' >> ${_dir}/utt2spk
        for f in text wav.scp; do
            <${src_dir}/${f} awk -v prefix="${prefix}" '{print(prefix $0)}' >> ${_dir}/${f}
        done
    done

    # ml-superb
    prefix="ml_suprb-"
    for dset in "train_1h" "dev_1h"; do
        if [ ${dset} = "train_1h" ]; then
            _dir="data/${train_set}"
        else
            _dir="data/${train_dev}"
        fi

        src_dir="data/ml_superb/${dset}"
        <${src_dir}/utt2spk awk -v prefix="${prefix}" '{print(prefix $1, prefix $2)}' >> ${_dir}/utt2spk
        for f in text wav.scp; do
            <${src_dir}/${f} awk -v prefix="${prefix}" '{print(prefix $0)}' >> ${_dir}/${f}
        done
    done

    for dset in train dev; do
        utils/utt2spk_to_spk2utt.pl data/${dset}/utt2spk > data/${dset}/spk2utt
        utils/fix_data_dir.sh data/${dset}
    done
fi
