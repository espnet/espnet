#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0

# general configuration
stage=1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)


# bpemode (unigram or bpe)
nbpe=30
bpemode=unigram

. utils/parse_options.sh || exit 1;
. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"

    # TODO(kamo): Change Makefile to install spk2pipe
    local/data.sh --sph2pipe ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: "
    steps/asr/prepare_token.sh \
        --nbpe "${nbpe}" --bpemode "${bpemode}" \
        data/train_nodev data/train_dev

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LM Preparation"
    log "Not yet"

    steps/lm/train.sh
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Network Training"
    steps/asr/train.sh --cmd "${cuda_cmd}" data/train_nodev data/train_dev

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Decoding"
    log "Not yet"
    exit 1

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    for dset in ${eval_sets}; do
        steps/asr/decode.sh --cmd "${_cmd} --gpu ${ngpu}" --nj "${decode_nj}" --ngpu "${ngpu}" ${expdir}/results data/${dset}
    done

    steps/asr/show_result.sh ${expdir}

fi
