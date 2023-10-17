#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# Centralized data preparation for OWSM (https://arxiv.org/abs/2309.13876)

# Note (jinchuan): 
# (1) please work progressively from v1 to v3: you need to 
# prepare data for v1, v2 and v3 in order to obtain the full v3 data
# (2) please revise db.sh for all datasets before running this script.
# Some datasets cannot be downloaded and untared automatically due to
# liscence issue. Please take care of it in advance.
# (3) Due to the large volume of data, we can not ensure the scripts
# will run smoothly for each dataset. Please raise an issue if you 
# believe there is a bug.
# (4) This script only prepare data for train and valid. Test data
# should be prepared separately following standard Espnet2 format.

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./db.sh || exit 1;

function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

VERSION=v1 # specify v1, v2 or v3
stage=1
stop_stage=2

. utils/parse_options.sh

# Change accordingly if you only want to prepare a subset of it
if [ ${VERSION} = "v1" ]; then
    datasets="aishell covost2 gigaspeech librispeech must-c spgispeech"
    train_sets="data/AISHELL-1/train \
                data/CoVoST2/train \
                data/GigaSpeech/XL \
                data/LibriSpeech/train-clean-100 \
                data/LibriSpeech/train-clean-360 \
                data/LibriSpeech/train-other-500 \
                data/MuST-C_v1.2/train \
                data/MuST-C_v2/train \
                data/MuST-C_v3/train \
                data/SPGISpeech/train \
                data/TEDLIUM3/train"
    valid_sets="data/AISHELL-1/dev \
                data/CoVoST2/dev \
                data/GigaSpeech/DEV \
                data/LibriSpeech/dev-clean \
                data/LibriSpeech/dev-other \
                data/MuST-C_v1.2/dev \
                data/MuST-C_v2/dev \
                data/MuST-C_v3/dev \
                data/SPGISpeech/val \
                data/TEDLIUM3/dev"

elif [ ${VERSION} = "v2" ]; then
    datasets="gigast multilingual_librispeech wenetspeech"
    train_sets="data/GigaST/XL.en-* \
                data/MLS/train.* \
                data/WenetSpeech/L"
    # question (jinchuan): why don't include GigaST-dev?
    valid_sets="data/MLS/dev.* \
                data/WenetSpeech/DEV"

elif [ ${VERSION} = "v3" ]; then
    datasets="aidatatang ami babel commonvoice swbd fisher_callhome \
              fleurs ksponspeech magicdata reazonspeech ru_open_stt \
              vctk voxpopuli wsj" \
    # still working on it
    train_sets="data/aidatatang/train \
    "
    valid_sets="data/aidatatang/dev \
    "
else
    echo "Invalid version argument ${VERSION}." && exit 1;
fi
echo "Preparing data for OSWM with version ${VERSION}"
echo "Datasets to prepare: ${datasets}"

utt_extra_files="text.prev text.ctc"
train_out=data/train_${VERSION}
valid_out=data/valid_${VERSION}

# call data preparation script for each dataset
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dataset in ${datasets}; do
        if [ -f data/.${dataset}.done ]; then
            echo ${dataset} has been processed. Skip!
        else
            echo preparing ${dataset} dataset ...
            ./local/prepare_${dataset}.sh && touch data/.${dataset}.done
        fi
    done
fi

# combine all datasets.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    if [ ${VERSION} = "v2" ]; then
        if [ ! -d data/train_v1 ] || [ ! -d data/valid_v1 ]; then
            echo "Cannot find v1 data. Please link it here. Exit!" && exit 1;
        fi
        train_sets="${train_sets} data/train_v1"
        valid_sets="${valid_sets} data/valid_v1"
    fi

    if [ ${VERSION} = "v3" ]; then
        if [ ! -d data/train_v2 ] || [ ! -d data/valid_v2 ]; then
            echo "Cannot find v2 data. Please link it here. Exit!" && exit 1;
        fi
        train_sets="${train_sets} data/train_v2"
        valid_sets="${valid_sets} data/valid_v2"
    fi

    # Combine valid
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        ${valid_out} ${valid_sets} || exit 1;
    # NOTE(yifan): extra text files must be sorted and unique
    for f in ${utt_extra_files}; do
        check_sorted ${valid_out}/${f}
    done
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${valid_out} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print ${valid_out} || exit 1;

    # Combine train
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        ${train_out} ${train_sets} || exit 1;
    # NOTE(yifan): extra text files must be sorted and unique
    for f in ${utt_extra_files}; do
        check_sorted ${train_out}/${f}
    done
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${train_out} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print ${train_out} || exit 1;
fi

# todo: some v3-specific operations


log "Successfully finished. [elapsed=${SECONDS}s]"
