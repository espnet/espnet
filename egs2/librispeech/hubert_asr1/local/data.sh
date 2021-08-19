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
pretrain_train_set="train_960"
pretrain_valid_set="dev"

finetune_train_set="train_10h"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for Librispeech Data"
    local/ls_data.sh \
	--train_set ${pretrain_train_set} \
	--train_dev ${pretrain_valid_set}
fi

if [ ${stage} -le 2 ] && [ ${stage} -ge 2 ]; then
    log "Stage 2: Data preparation for Librilight Data"
    ./local/prepare_librilight.sh \
	--train_set ${finetune_train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
