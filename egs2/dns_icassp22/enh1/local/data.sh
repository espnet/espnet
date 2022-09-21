#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] [--total_hours <total_hours>] [--nj <nj>]
  optional argument:
    [--stage]: 0 (default), or 1, or 2
    [--stop_stage]: 0, or 1, or 2 (default)
    [--total_hours]: amount (in hours) of synthetic noisy speech to generate (default=150, DNS4-en read speech has ~600 hours, 
                     set this somewhat below that to avoid duplicate data)
    [--nj] number of jobs created to synthesize noisy data (default=1)
    NOTE:
        stage 0: Download dataset with the script from DNS-Challenge official repo
        stage 1: Create the Data Mixture from the DNS scripts. You can skip this step when you already have the audio mixture for training.
        stage 2: Prepare the data for ESPnet-se
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=0
stop_stage=2
total_hours=150
nj=1

dns_wav=${DNS4}/dns_wav

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "${DNS4}" ]; then
    log "Fill the value of 'DNS4' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data download"
    local/download_dns4_dataset.sh ${DNS4} || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data simulation: mix clean speech with noise clips"
    local/dns_create_mixture.sh ${DNS4} ${dns_wav} ${total_hours} ${nj} || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    # The following datasets will be created:
    # {tr,cv,tt}_synthetic
    local/dns_data_prep.sh  ${dns_wav} || exit 1;
fi
