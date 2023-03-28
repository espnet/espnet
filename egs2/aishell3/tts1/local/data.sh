#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=3
threshold=35
nj=40

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${AISHELL3}" ]; then
   log "Fill the value of 'AISHELL3' of db.sh"
   exit 1
fi
db_root=${AISHELL3}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage -1: download data from openslr"
    local/download_and_untar.sh "${db_root}" "https://www.openslr.org/resources/93/data_aishell3.tgz" data_aishell3.tgz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: prepare aishell3 data"
    mkdir -p data
    for x in train test; do
        mkdir -p data/${x}
        python local/data_prep.py --src "${db_root}"/${x}/ --dest data/${x}
        sort data/${x}/utt2spk -o data/${x}/utt2spk
        sort data/${x}/wav.scp -o data/${x}/wav.scp
        sort data/${x}/text -o data/${x}/text
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/validate_data_dir.sh --no-feats data/${x}
    done

    for x in train_phn test_phn; do
        mkdir -p data/${x}
        python local/data_prep.py --src "${db_root}"/"$(echo ${x} | cut -d'_' -f 1)"/ --dest data/${x} --external_g2p false
        sort data/${x}/utt2spk -o data/${x}/utt2spk
        sort data/${x}/wav.scp -o data/${x}/wav.scp
        sort data/${x}/text -o data/${x}/text
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/validate_data_dir.sh --no-feats data/${x}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: scripts/audio/trim_silence.sh"
    for x in train test train_phn test_phn; do
        # shellcheck disable=SC2154
        scripts/audio/trim_silence.sh \
             --cmd "${train_cmd}" \
             --nj "${nj}" \
             --fs 44100 \
             --win_length 2048 \
             --shift_length 512 \
             --threshold "${threshold}" \
             data/${x} data/${x}/log

        utils/fix_data_dir.sh data/${x}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: split for development set"
    utils/subset_data_dir.sh data/train 250 data/dev
    utils/subset_data_dir.sh data/train_phn 250 data/dev_phn
    utils/copy_data_dir.sh data/train data/train_no_dev
    utils/copy_data_dir.sh data/train_phn data/train_phn_no_dev
    utils/filter_scp.pl --exclude data/dev/wav.scp \
        data/train/wav.scp > data/train_no_dev/wav.scp
    utils/filter_scp.pl --exclude data/dev_phn/wav.scp \
        data/train_phn/wav.scp > data/train_phn_no_dev/wav.scp
    utils/fix_data_dir.sh data/train_no_dev
    utils/fix_data_dir.sh data/train_phn_no_dev
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
