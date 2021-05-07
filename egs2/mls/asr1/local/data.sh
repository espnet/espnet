#!/usr/bin/env bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang=es


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

log "data preparation started"

if [ -z "${MLS}" ]; then
    log "Fill the value of 'MLS' of db.sh"
    exit 1
fi


# Get lang's name from its id for download links
case ${lang} in
    "es")
        download_id=spanish ;;
    "en")
        download_id=english ;;
    "fr")
        download_id=french ;;
    "nl")
        download_id=dutch ;;
    "it")
        download_id=italian ;;
    "pt")
        download_id=portuguese ;;
    "pl")
        download_id=polish ;;
esac

data_url=https://dl.fbaipublicfiles.com/mls/mls_${download_id}.tar.gz
lm_data_url=https://dl.fbaipublicfiles.com/mls/mls_lm_${download_id}.tar.gz

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${MLS}"

    local/download_and_untar.sh ${MLS} ${data_url} mls_${download_id}.tar.gz
    local/download_and_untar.sh ${MLS} ${lm_data_url} mls_lm_${download_id}.tar.gz
    # Optional: mls corpus is large. You might want to remove them after processing
    # rm -f ${MLS}/mls_${download_id}.tar.gz
    # rm -f ${MLS}/mls_lm_${download_id}.tar.gz
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for MLS"

    python local/data_prep.py --source ${MLS}/mls_${download_id} --lang ${lang} --prefix "mls_"
    utils/fix_data_dir.sh data/${lang}_train
    utils/fix_data_dir.sh data/${lang}_dev
    utils/fix_data_dir.sh data/${lang}_test

    # add placeholder to align format with other corpora
    sed -r '/^\s*$/d' ${MLS}/mls_lm_${download_id}/data.txt | \
         awk '{printf("%.8d %s\n"), NR-1, $0}'  > data/${lang}_lm_train.txt
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
