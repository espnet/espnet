#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
splits_dir=data/iwslt23_splits

log "$0 $*"
. utils/parse_options.sh

if [ -z "${IWSLT23_LOW_RESOURCE}" ]; then
    log "Fill the value of 'IWSLT23_LOW_RESOURCE' of db.sh"
    exit 1
fi

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${splits_dir}" ]; then
    log "stage 1: Official splits from IWSLT"
    
    git clone https://github.com/Llamacha/IWSLT2023_Quechua_data.git ${splits_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    
    mkdir -p data/train
    mkdir -p data/valid
    mkdir -p data/test
    python local/preprocess.py --in_path ${splits_dir}/que_spa_clean
    
    for set in train valid "test"
    do
        cp data/${set}/text.spa data/${set}/text
        utils/utt2spk_to_spk2utt.pl data/${set}/utt2spk > data/${set}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "text.que text.spa" data/${set}
        utils/validate_data_dir.sh --no-feats data/${set} || exit 1
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Normalize Transcripts"

    # check extra module installation
    if ! command -v tokenizer.perl > /dev/null; then
        echo "Error: it seems that moses is not installed." >&2
        echo "Error: please install moses as follows." >&2
        echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
        exit 1
    fi

    for set in train valid "test"
    do
        cut -d ' ' -f 2- data/${set}/text.que > data/${set}/que.org
        cut -d ' ' -f 1 data/${set}/text.que > data/${set}/uttlist
        # remove punctuation
        remove_punctuation.pl < data/${set}/que.org > data/${set}/que.rm
        paste -d ' ' data/${set}/uttlist data/${set}/que.rm > data/${set}/text.lc.rm.que

        cut -d ' ' -f 2- data/${set}/text.spa > data/${set}/spa.org
        # tokenize
        tokenizer.perl -l es -q < data/${set}/spa.org > data/${set}/spa.tok
        paste -d ' ' data/${set}/uttlist data/${set}/spa.tok > data/${set}/text.tc.spa

        # remove empty lines that were previously only punctuation
        # small to use fix_data_dir as is, where it does reduce lines based on extra files
        <"data/${set}/text.lc.rm.que" awk ' { if( NF != 1 ) print $0; } ' >"data/${set}/text"
        utils/fix_data_dir.sh --utt_extra_files "text.lc.rm.que text.tc.spa text.spa text.que" data/${set}
        cp data/${set}/text.tc.spa data/${set}/text
        utils/fix_data_dir.sh --utt_extra_files "text.lc.rm.que text.tc.spa text.spa text.que" data/${set}
        utils/validate_data_dir.sh --no-feats data/${set} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
