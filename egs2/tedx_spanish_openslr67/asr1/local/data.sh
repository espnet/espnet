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

log "$0 $*"
. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${TEDX_SPANISH}" ]; then
  log "Error: \$TEDX_SPANISH is not set in db.sh."
  exit 2
fi

# download dataset
log "Download data to ${TEDX_SPANISH}"
if [ ! -d "${TEDX_SPANISH}" ]; then
    mkdir -p "${TEDX_SPANISH}"
    (
    cd ${TEDX_SPANISH}
    wget "https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz"
    tar -xvf tedx_spanish_corpus.tgz
    rm tedx_spanish_corpus.tgz
    git clone https://github.com/jessicah25/tedx_spanish_corpus_split.git ../local/split
    cd ..
    )
else
    echo "data is already downloaded."
fi


mkdir -p data/train
mkdir -p data/dev
mkdir -p data/test

export TEDX_SPANISH

# create train, dev, test split 90/5/5
# python local/split_data.py

# create spk2utt, utt2spk, wav.scp files for recipe
python local/create_data.py

utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

./utils/fix_data_dir.sh data/train/
./utils/fix_data_dir.sh data/dev/
./utils/fix_data_dir.sh data/test/

log "Successfully finished. [elapsed=${SECONDS}s]"
