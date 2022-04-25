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


stage=1
stop_stage=100000
fs=16k
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${HARPERVALLEY}" ]; then
    log "Fill the value of 'HARPERVALLEY' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${HARPERVALLEY}/LICENSE" ]; then
        echo "stage 1: Download data to ${HARPERVALLEY}"
        mkdir -p ${HARPERVALLEY}
        git clone https://github.com/cricketclub/gridspace-stanford-harper-valley.git ${HARPERVALLEY}
    else
        log "stage 1: ${HARPERVALLEY}/LICENSE is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2 : Data Preparation"
    if [ -n "$(ls data/tmp/)" ]; then
        rm -r data/tmp/
    fi
    for file in ${HARPERVALLEY}/data/transcript/*.json; do
        filename=$(basename "${file%.*}")
        dirname="${HARPERVALLEY}/data/"
        python3 local/data_prep.py --source_dir "$dirname" \
            --audio_dir "data/audio" \
            --filename "$filename" \
            --target_dir "data/tmp" \
            --min_length 4
    done
    sed -i -e 's/<unk>/\[unk\]/g' data/tmp/text
    mkdir -p data/{train,valid,test}
    python3 local/split_data.py --source_dir "data/tmp" \
        --min_spk_utt 10 \
        --train_frac 0.8 \
        --val_frac 0.1
    for x in test valid train; do
        for f in text wav.scp utt2spk segments; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
