#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./utils/parse_options.sh

log "data preparation started"

if [ -z "${FSD50K}" ]; then
    log "Fill the value of 'FSD50K' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${FSD50K}"

    # Downloading from zenodo records
    # https://zenodo.org/records/4060432
    fnames=(
        "FSD50K.dev_audio.z01"
        "FSD50K.dev_audio.z02"
        "FSD50K.dev_audio.z03"
        "FSD50K.dev_audio.z04"
        "FSD50K.dev_audio.z05"
        "FSD50K.dev_audio.zip"
        "FSD50K.doc.zip"
        "FSD50K.eval_audio.z01"
        "FSD50K.eval_audio.zip"
        "FSD50K.metadata.zip"
    )
    for fname in "${fnames[@]}"; do
        wget -O ${FSD50K}/${fname} \
            https://zenodo.org/records/4060432/files/${fname}?download=1       
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Unzip data in ${FSD50K}"

    zip -s 0 ${FSD50K}/FSD50K.dev_audio.zip --out ${FSD50K}/dev.zip
    zip -s 0 ${FSD50K}/FSD50K.eval_audio.zip --out ${FSD50K}/eval.zip

    fnames=(
        "dev.zip"
        "eval.zip"
        "FSD50K.metadata.zip"
        "FSD50K.doc.zip"
    )
    for fname in "${fnames[@]}"; do
        unzip ${FSD50K}/${fname} -d ${FSD50K}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare data in ${FSD50K}"

    python3 local/prep_fsd50k.py \
        --train ${FSD50K}/FSD50K.dev_audio \
        --test ${FSD50K}/FSD50K.eval_audio \
        --metadata ${FSD50K}/FSD50K.ground_truth \
        --output data
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Validate data"

    for x in train val test; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
