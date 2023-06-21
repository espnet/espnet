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
splits_dir=data/iwslt22_splits

log "$0 $*"
. utils/parse_options.sh

if [ -z "${IWSLT22_DIALECT}" ]; then
    log "Fill the value of 'IWSLT22_DIALECT' of db.sh"
    exit 1
fi

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${splits_dir}" ]; then
    log "stage 1: Official splits from IWSLT"
    
    git clone https://github.com/kevinduh/iwslt22-dialect.git ${splits_dir}
    cd ${splits_dir} && ./setup_data.sh ${IWSLT22_DIALECT} && cd -
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    
    mkdir -p data/train
    mkdir -p data/dev
    mkdir -p data/test1
    local/preprocess.py --out data --data ${splits_dir}
    
    for set in train dev test1
    do
        cp data/${set}/text.en data/${set}/text
        utils/utt2spk_to_spk2utt.pl data/${set}/utt2spk > data/${set}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "text.en text.ta" data/${set}
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

    for set in train dev test1
    do
        cut -d ' ' -f 2- data/${set}/text.ta > data/${set}/ta.org
        cut -d ' ' -f 1 data/${set}/text.ta > data/${set}/uttlist
        # remove punctuation
        remove_punctuation.pl < data/${set}/ta.org > data/${set}/ta.rm
        paste -d ' ' data/${set}/uttlist data/${set}/ta.rm > data/${set}/text.tc.rm.ta

        cut -d ' ' -f 2- data/${set}/text.en > data/${set}/en.org
        # tokenize
        tokenizer.perl -l en -q < data/${set}/en.org > data/${set}/en.tok
        paste -d ' ' data/${set}/uttlist data/${set}/en.tok > data/${set}/text.tc.en

        # remove empty lines that were previously only punctuation
        # small to use fix_data_dir as is, where it does reduce lines based on extra files
        <"data/${set}/text.tc.rm.ta" awk ' { if( NF != 1 ) print $0; } ' >"data/${set}/text"
        utils/fix_data_dir.sh --utt_extra_files "text.tc.rm.ta text.tc.en text.en text.ta" data/${set}
        cp data/${set}/text.tc.en data/${set}/text
        utils/fix_data_dir.sh --utt_extra_files "text.tc.rm.ta text.tc.en text.en text.ta" data/${set}
        utils/validate_data_dir.sh --no-feats data/${set} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
