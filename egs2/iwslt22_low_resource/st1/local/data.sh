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
splits_dir=data/iwslt22_taq_fra_split

log "$0 $*"
. utils/parse_options.sh

if [ -z "${IWSLT22_LOW_RESOURCE}" ]; then
    log "Fill the value of 'IWSLT22_LOW_RESOURCE' of db.sh"
    exit 1
fi

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${splits_dir}" ]; then
    log "stage 1: Official splits from IWSLT Low Resource Speech Translation"

    git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git ${splits_dir}

    # train comprises 17 hours of clean speech in Tamasheq, translated to the French language
    # train_full comprises a 19 hour version of this corpus,
    # including 2 additional hours of data that was labeled by annotators as potentially noisy
    mkdir -p data/train/org
    mkdir -p data/train_full/org
    mkdir -p data/valid/org
    mkdir -p data/test/org

    for set in train valid test
    do
        cp -r ${splits_dir}/taq_fra_clean/${set}/* data/${set}/org
    done
    cp -r ${splits_dir}/taq_fra_full/train/* data/train_full/org

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    
    for set in train train_full valid test
    do
        python local/preprocess.py --out data/${set} --data data/${set}/org

        cp data/${set}/text.fr data/${set}/text

        utils/utt2spk_to_spk2utt.pl data/${set}/utt2spk > data/${set}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "text.fr" data/${set}
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

    for set in train train_full valid test
    do
        cut -d ' ' -f 2- data/${set}/text.fr > data/${set}/fr.org
        cut -d ' ' -f 1 data/${set}/text.fr > data/${set}/uttlist

        # tokenize
        tokenizer.perl -l fr -q < data/${set}/fr.org > data/${set}/fr.tok
        paste -d ' ' data/${set}/uttlist data/${set}/fr.tok > data/${set}/text.tc.fr

        # remove empty lines that were previously only punctuation
        # small to use fix_data_dir as is, where it does reduce lines based on extra files
        <"data/${set}/text.tc.fr" awk ' { if( NF != 1 ) print $0; } ' >"data/${set}/text"
        utils/fix_data_dir.sh --utt_extra_files "text.tc.fr text.fr" data/${set}
        cp data/${set}/text.tc.fr data/${set}/text
        utils/fix_data_dir.sh --utt_extra_files "text.tc.fr text.fr" data/${set}
        utils/validate_data_dir.sh --no-feats data/${set} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
