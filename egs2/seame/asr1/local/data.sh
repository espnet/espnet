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
dev_repo_dir=data/SEAME-dev-set

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${SEAME}" ]; then
    log "Fill the value of 'SEAME' of db.sh"
    exit 1
fi

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -d "${dev_repo_dir}" ]; then
    log "stage 1: Clone official SEAME repository"
    
    git clone https://github.com/zengzp0912/SEAME-dev-set.git ${dev_repo_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    
    local/preprocess.py --out data --data ${SEAME} --repo ${dev_repo_dir}
    
    for set in train valid devman devsge
    do
        cp data/${set}/text.rm.noise data/${set}/text
        utils/utt2spk_to_spk2utt.pl data/${set}/utt2spk > data/${set}/spk2utt
        utils/validate_data_dir.sh --no-feats data/${set} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
