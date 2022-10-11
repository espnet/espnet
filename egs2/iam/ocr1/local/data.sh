#!/bin/bash
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

. ./db.sh
. ./path.sh
. ./cmd.sh

stage=1
stop_stage=2

# Fill in username/password from account on https://fki.tic.heia-fr.ch/register
iam_username=""
iam_password=""

# Set parameters for the feature dimensions used during image extraction,
# see data_prep.py for details
feature_dim=100
downsampling_factor=0.5

data_dir=data/

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1.1: Downloading the IAM Handwriting dataset with username ${iam_username} and password ${iam_password}"
    mkdir -p ${IAM}
    local/download_and_untar.sh ${IAM} ${iam_username} ${iam_password}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 1.2: Data Preparation - generating text, utt2spk, spk2utt, and feats.scp for train/valid/test splits"
    if [ -e ${data_dir} ]; then
        echo "Directory ${data_dir} already exists, removing it"
        rm -rf ${data_dir}
    fi
    python local/data_prep.py --feature_dim ${feature_dim} --downsampling_factor ${downsampling_factor}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"