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
stop_stage=100


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi


if [ -z "${EDACC}" ]; then
    log "Fill the value of 'EDACC' of db.sh"
    exit 1
fi

partitions="dev test"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${EDACC}/edacc_v1.0/README.txt" ]; then
        echo "stage 1: Please download data from https://datashare.ed.ac.uk/handle/10283/4766 and save to ${EDACC}"
    else
        log "stage 1: ${EDACC}/edacc_v1.0/README.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    
    # prepare the date in Kaldi style
    python3 local/data_prep.py "${EDACC}/edacc_v1.0" "data"

    # sort the data, and make utt2spk to spk2utt
    for x in ${partitions}; do
        for f in text wav.scp utt2spk segments; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done

    # # make a train set (ask for guidance of it is necessary)

    # Validate data
    for x in ${partitions}; do
        utils/validate_data_dir.sh --no-feats "data/${x}" 
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
