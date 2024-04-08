#!/usr/bin/env bash

# Copyright 2021 Yuekai Zhang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=3
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

log "data preparation started"

if [ -z "${SNIPS}" ]; then
    log "Fill the value of 'SNIPS' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to"
    local/download_and_untar.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for snips"
    mkdir -p data
    python local/data_prep.py --wav_path $SNIPS
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   log "stage 2: Processing the text files"
   mkdir -p data/tmp
   sort -u data/text.trans > data/tmp/text.sort
   sort  -o data/wav.scp  data/wav.scp
   sort -uo data/non_linguistic_symbols.txt data/non_linguistic_symbols.txt
   sort -u data/semantics > data/tmp/semantics.sort
   rm -r data/semantics
   awk '{print $(NF)}' data/tmp/text.sort > data/tmp/wavs
   paste -d "|" data/tmp/wavs data/tmp/semantics.sort > data/tmp/wav_sem_text
   cut -d "|" -f 1,3 data/tmp/wav_sem_text > data/tmp/wav_sem
   cut -d "|" -f 2 data/tmp/wav_sem_text > data/tmp/pure_text
   tr '[:upper:]' '[:lower:]' <data/tmp/pure_text>data/tmp/pure_text_lower
   tr -d '[:punct:]'< data/tmp/pure_text_lower > data/tmp/pure_text_norm
   sed -r 's/\|/ /g' data/tmp/wav_sem > data/tmp/wav_sem.2
   cut -f 1,2 -d " " data/tmp/wav_sem.2  > data/tmp/wav_intent
   paste -d " " data/tmp/wav_intent data/tmp/pure_text_norm > data/text
   mkdir -p data/all
   mv data/utt2spk data/text data/wav.scp data/all
   utils/fix_data_dir.sh data/all
   rm -r data/tmp
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   utils/subset_data_dir.sh --first data/all 332 data/dev_test
   n=$(($(wc -l < data/all/text) - 332))
   utils/subset_data_dir.sh --last data/all ${n} data/train
   utils/fix_data_dir.sh data/train
   utils/subset_data_dir.sh --first data/dev_test 166 data/dev
   n=$(($(wc -l < data/dev_test/text) - 166))
   utils/subset_data_dir.sh --last data/dev_test ${n} data/test
   utils/fix_data_dir.sh data/dev
   utils/fix_data_dir.sh data/test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
