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
ASVSpoof_CMD=/scratch/bbjs/shared/corpora/esc_master/ESC-50-master

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${ASVSpoof_CMD}" ]; then
    log "Fill the value of 'ASVSpoof' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${ASVSpoof_CMD}/LICENSE" ]; then
	    echo "stage 1: Download data to ${ASVSpoof_CMD}"
        # git clone https://github.com/kolesov93/lt_speech_commands.git ${LT_SPEECH_CMD}
    else
        log "stage 1: ${ASVSpoof_CMD}/LICENSE is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/data_prep.py ${ASVSpoof_CMD}
    for x in test valid train; do
        for f in text wav.scp utt2spk; do
            echo ${x}, ${f}
            echo "?"
            head data/${x}/${f}
            sort data/${x}/${f} -o data/${x}/${f}
            echo "====="
            head data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
