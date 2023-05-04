#!/usr/bin/env bash
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=0
SECONDS=0

. utils/parse_options.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -e "${NSC}" ]; then
    log "Fill the value of 'NSC' of db.sh"
    exit 1
fi

# Use Lhotse to prepare the data dir
# copied from https://github.com/pzelasko/kaldi/blob/feature/nsc-recipe/egs/nsc/s5/local/nsc_data_prep.sh

# Pre-requisites
pip install lhotse
pip install git+https://github.com/pzelasko/Praat-textgrids

if [ $stage -le 0 ]; then
    lhotse prepare nsc ${NSC} data/nsc
    lhotse kaldi export data/nsc/recordings_PART3_SameCloseMic.json data/nsc/supervisions_PART3_SameCloseMic.json data/nsc
    utils/fix_data_dir.sh data/nsc
    utils/utt2spk_to_spk2utt.pl data/nsc/utt2spk > data/nsc/spk2utt
    # "Poor man's text normalization"
    mv data/nsc/text data/nsc/text.bak
    sed 's/[#!~()*]\+//g' data/nsc/text.bak \
        | sed 's/<UNK>/XPLACEHOLDERX/g' \
        | sed 's/<.\+>//g' \
        | sed 's/XPLACEHOLDERX/<UNK>/g' \
        > data/nsc/text

    # Create a train and test splits by following
    # https://github.com/pzelasko/kaldi/blob/feature/nsc-recipe/egs/nsc/s5/local/nsc_data_prep.sh#L30-L35
    n_spk=$(wc -l data/nsc/spk2utt | cut -f1 -d' ')
    tail -10 data/nsc/spk2utt | cut -f1 -d' ' > data/test.spk
    head -n $((n_spk - 10)) data/nsc/spk2utt | cut -f1 -d' ' > data/train.spk
    utils/subset_data_dir.sh --spk-list data/train.spk data/nsc data/train
    utils/subset_data_dir.sh --spk-list data/test.spk data/nsc data/test

    # Make a dev set
    utils/subset_data_dir.sh --first data/train 4000 data/dev
    n=$(($(wc -l < data/train/text) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

    # Remove temp files
    rm -f data/*.spk

    # FIXME(sw005320): reco2dur is not properly handled in the current script
    # I just removed them
    rm data/*/reco2dur
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
