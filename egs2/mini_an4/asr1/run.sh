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
stage=1
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)

nj=50

audio_format=flac

# bpemode (unigram or bpe)
nbpe=30
bpemode=unigram

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh


train_set=train_nodev
dev_set=train_dev
eval_sets=test


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"

    # TODO(kamo): Change Makefile to install sph2pipe
    local/data.sh --sph2pipe ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Format wav.scp"
    for dset in ${train_set} ${dev_set} ${eval_sets}; do
        utils/copy_data_dir.sh data/${dset} data_format/${dset}
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format ${audio_format} \
            data/${dset}/wav.scp data_format/${dset}
    done

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Generate tokens from text"
    scripts/asr/prepare_token.sh \
        --nbpe "${nbpe}" --bpemode "${bpemode}" \
            data_format/${train_set} data_format/${dev_set}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LM Preparation"
    if false; then
        scripts/lm/train.sh --cmd "${cuda_cmd} --gpu ${ngpu}" --ngpu "${ngpu}" \
            data_format/${train_set}_bpe_${train_set}_${bpemode}${nbpe}/text \
            data_format/${dev_set}_bpe_${train_set}_${bpemode}${nbpe}/text \
            exp/lm_train
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Network Training"
    scripts/asr/train.sh --cmd "${cuda_cmd} --gpu ${ngpu}" --ngpu "${ngpu}" \
        data_format/${train_set}_bpe_${train_set}_${bpemode}${nbpe} \
        data_format/${dev_set}_bpe_${train_set}_${bpemode}${nbpe} \
        exp/asr_train
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
        scripts/asr/decode.sh --cmd "${_cmd} --gpu ${ngpu}" --nj "${decode_nj}" --ngpu "${ngpu}" \
            ${expdir}/results data/${dset} ${expdir}/decode_${dset}
    done

    scripts/asr/show_result.sh ${expdir}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
