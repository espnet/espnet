#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>]
  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
    NOTE:
        stage 1: Create the Data Mixture from the DNS scripts. You can skip this step when you already have the audio mixture for training.
        stage 2: Prepare the data for ESPNet-se
        You can get scripts by git clone -b icassp2021-final https://github.com/microsoft/DNS-Challenge.git DNS-Challenge
        You can download the data by using download-dns-challenge-2.sh in the master branch without git lfs
        In addition, "datasets/wideband/acoustic_params_wideband" and "datasets/wideband/dev_testset_wideband/track1" are required, which are not downloaded by the above script
        You can find them in interspeech/adddata branch
        For evaluation, synthetic data in the "datasets/wideband/dev_testset_wideband/track1" in the the interspeech2021/adddata branch is used
        To avoid issues related to hard-coded paths, please change the current directory to DNS-Challenge in noisyspeech_synthesizer_singleprocess.py
        Also, please make sure the destination is under data/dns_wav
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=2
dns_wav=$PWD/data/dns_wav


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${DNS2}" ]; then
    log "Fill the value of 'DNS2' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"
    local/dns_create_mixture.sh ${DNS2} ${dns_wav}  || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    # The following datasets will be created:
    # {tr,cv}_synthetic tt_synthetic_track_1
    local/dns_data_prep.sh  ${dns_wav} ${DNS2}/datasets/dev_testset_wideband || exit 1;
fi
