#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0
  optional argument:
    None
EOF
)

. ./path.sh
. ./db.sh


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${NOISY_REVERBERANT_SPEECH}" ] || [ ! -e "${NOISY_SPEECH}" ] ; then
    log "
    Please fill the value of 'NOISY_REVERBERANT_SPEECH' and 'NOISY_SPEECH' in db.sh
    The 'NOISY_REVERBERANT_SPEECH' (https://doi.org/10.7488/ds/2139)
    directory should be like:
        noisy_reverberant_speech
        ├── logfiles
        ├── noisyreverb_testset_wav
        ├── noisyreverb_trainset_28spk_wav
        └── noisyreverb_trainset_56spk_wav
    the 'NOISY_SPEECH' (https://doi.org/10.7488/ds/2117) directory
    should at least contain the clean reference:
        noisy_speech
        ├── clean_testset_wav
        ├── clean_trainset_28spk_wav
        └── clean_trainset_56spk_wav
    "
    exit 1
fi



for dset in testset trainset_28spk trainset_56spk;
do

  mkdir -p data/${dset}
  awk '{print $1 " '${NOISY_REVERBERANT_SPEECH}/noisyreverb_${dset}_wav/'"$1".wav"}' ${NOISY_REVERBERANT_SPEECH}/logfiles/log_${dset}.txt | sort -n  > data/${dset}/wav.scp
  awk '{print $1 " '${NOISY_SPEECH}/clean_${dset}_wav/'"$1".wav"}' ${NOISY_REVERBERANT_SPEECH}/logfiles/log_${dset}.txt | sort -n  > data/${dset}/spk1.scp
  awk -F '_| ' '{print $1"_"$2, $1}' ${NOISY_REVERBERANT_SPEECH}/logfiles/log_${dset}.txt | sort -n  > data/${dset}/utt2spk
  ./utils/utt2spk_to_spk2utt.pl  data/${dset}/utt2spk >  data/${dset}/spk2utt
done

# By default, combine the 28spk and the 56spk dataset together
combine_data.sh --extra-files 'spk1.scp' data/train_28_and_56spk data/trainset_28spk data/trainset_56spk

# Split the whole training set into train and valid
./utils/subset_data_dir.sh --spk-list local/train_spk data/train_28_and_56spk data/train
./utils/fix_data_dir.sh data/train
./utils/filter_scp.pl data/train/wav.scp data/train_28_and_56spk/spk1.scp > data/train/spk1.scp

./utils/subset_data_dir.sh --spk-list local/valid_spk data/train_28_and_56spk data/valid
./utils/fix_data_dir.sh data/valid
./utils/filter_scp.pl data/valid/wav.scp data/train_28_and_56spk/spk1.scp > data/valid/spk1.scp
