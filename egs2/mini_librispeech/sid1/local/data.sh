#!/bin/bash

# Copyright 2023 Carnegie Mellon University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=1
SECONDS=0

mini_librispeech_url=http://www.openslr.org/resources/31


. ./utils/parse_options.sh || exit 1

if [ -z "${MINI_LIBRISPEECH}" ]; then
    log "Fill the value of 'MINI_LIBRISPEECH' of db.sh"
    exit 1
fi

mkdir -p "${MINI_LIBRISPEECH}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo " Stage 0: prepare data"

    #local/download_and_untar.sh "${MINI_LIBRISPEECH}" "${mini_librispeech_url}" train-clean-5

    if [ ! -f "${MINI_LIBRISPEECH}"/train_clean_5.done ]; then
        local/data_prep.sh "${MINI_LIBRISPEECH}"/LibriSpeech/train-clean-5 data/train_clean_5 || exit 1
        touch "${MINI_LIBRISPEECH}"/train_clean_5.done
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Prepare Kaldi-style train-test"
    ./utils/copy_data_dir.sh data/train_clean_5 data/train
    awk 'NR % 5 == 2 || NR % 5 == 1' data/train_clean_5/utt2spk > data/train/text
    ./utils/fix_data_dir.sh data/train

    ./utils/copy_data_dir.sh data/train_clean_5 data/dev
    awk 'NR % 5 == 3' data/train_clean_5/utt2spk > data/dev/text
    ./utils/fix_data_dir.sh data/dev

    ./utils/copy_data_dir.sh data/train_clean_5 data/test
    awk 'NR % 5 == 0' data/train_clean_5/utt2spk > data/test/text
    ./utils/fix_data_dir.sh data/test

fi
