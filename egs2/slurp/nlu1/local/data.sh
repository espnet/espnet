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
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SLURP}" ]; then
    log "Fill the value of 'SLURP' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${SLURP}/LICENSE.txt" ]; then
	echo "stage 1: Download data to ${SLURP}"
    else
        log "stage 1: ${SLURP}/LICENSE.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/prepare_slurp_data.py ${SLURP}
    for x in test devel train; do
        for f in text.ner text.intent text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done

    for x in test devel train; do
        cp data/${x}/text data/${x}/text.tc.en
        cp data/${x}/text.ner data/${x}/text.tc.ner
        cp data/${x}/text.intent data/${x}/text.tc.intent

        lowercase.perl < data/${x}/text > data/${x}/text.lc.en
        remove_punctuation.pl < data/${x}/text.lc.en > data/${x}/text.lc.rm.en
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
