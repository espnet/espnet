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
stop_stage=100000
flac2wav=false
nj=32

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh



if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${MYST}" ]; then
    log "Fill the value of 'MYST' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${MYST}/myst_child_conv_speech" ]; then
	    echo "stage 1: Please download data from https://catalog.ldc.upenn.edu/LDC2021S05 and save to ${MYST}"
        exit 1
    else
        log "stage 1: ${MYST}/myst_child_conv_speech is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if "${flac2wav}"; then
        log "stage 2: Convert flac to wav"
        original_dir="${MYST}/myst_child_conv_speech/data"
        logdir="${MYST}/myst_child_conv_speech/log"
        mkdir -p $logdir
        cmd=${train_cmd}
        ${cmd} "JOB=1:1" "${logdir}/flac_to_wav.JOB.log" \
            python local/flac_to_wav.py \
                --multiprocessing \
                --njobs ${nj} \
                --myst_dir ${original_dir}
    else
        log "flac2wav is false. Skip convertion."
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data preparation"
    # Set the base directory of the original dataset
    original_dir="${MYST}/myst_child_conv_speech/data"
    # Set the base directory for the target data
    data_dir="./data"

    if "${flac2wav}"; then
        python local/prepare_data.py --original-dir $original_dir --data-dir $data_dir --is-wav
    else
        python local/prepare_data.py --original-dir $original_dir --data-dir $data_dir
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
