#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# NOTE(Jinchuan):
# LibriLight is the audio file;
# LibriHeavy is the manifest information
# By running this script, we assume you already downloaded and unziped LibriLight.
if [ -z "${LIBRILIGHT}" ]; then
   log "Fill the value of 'LIBRILIGHT' of db.sh"
   exit 1
fi
if [ -z "${LIBRIHEAVY}" ]; then
   log "Fill the value of 'LIBRIHEAVY' of db.sh"
   exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Downloading manifests from huggingface."
    mkdir -p ${LIBRIHEAVY}
    for subset in small medium large test_clean test_other dev test_clean_large test_other_large; do
        if [ ! -e libriheavy_cuts_${subset}.jsonl.gz ]; then
        log "Downloading ${subset} subset."
        wget -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_${subset}.jsonl.gz -P ${LIBRIHEAVY}
        else
        log "Skipping download, ${subset} subset exists."
        fi
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Prepare kaldi style dataset"
    if [ ! -d libriheavy ]; then
        git clone https://github.com/k2-fsa/libriheavy.git
    fi
    mkdir -p data; mkdir -p data/local
    for subset in small medium large test_clean test_other dev test_clean_large test_other_large; do
        # parse the manifest
        python libriheavy/scripts/extract_and_normalize_transcript.py \
        --manifest ${LIBRIHEAVY}/libriheavy_cuts_${subset}.jsonl.gz \
        --subset ${subset} \
        --output-dir data/local/${subset}

        # wav.scp, segments, text
        mkdir -p data/${subset}
        cp data/local/${subset}/upper_no_punc/kaldi/${subset}/{text,segments} data/${subset}
        cat data/local/${subset}/upper_no_punc/kaldi/${subset}/wav.scp |\
            sed "s|download/librilight|${LIBRILIGHT}|" > data/${subset}/wav.scp

        # utt2spk, spk2utt
        cat data/${subset}/segments | awk '{print $1, $2}' > data/${subset}/utt2spk
        ./utils/fix_data_dir.sh data/${subset}
    done
fi
