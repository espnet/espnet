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
use_transcript=false
transcript_folder=
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${VOXCELEB}" ]; then
    log "Fill the value of 'VOXCELEB' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${VOXCELEB}/LICENSE.txt" ]; then
	echo "stage 1: Download data to ${VOXCELEB}"
    else
        log "stage 1: ${VOXCELEB}/LICENSE.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train,devel,test}
    if ${use_transcript}; then
        python3 local/data_prep_slue_transcript.py ${VOXCELEB} ${transcript_folder}
    else
        python3 local/data_prep_slue.py ${VOXCELEB}
    fi
    for x in test devel train; do
        for f in text wav.scp utt2spk transcript; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
    local/run_spm.sh
    mv data data_old
    mv data_bpe_1000 data
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
