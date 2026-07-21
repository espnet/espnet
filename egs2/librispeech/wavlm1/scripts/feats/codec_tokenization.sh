#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
nj=4                # number of parallel jobs
python=python3      # Specify python to execute espnet commands.
codec_choice=ESPnet # Options: Encodec, DAC, ESPnet (our in-house model)
codec_fs=16000
batch_size=3
bias=0
dump_audio=false
file_name=
src_dir=
tgt_dir=
checkpoint_path=null
config_path=null
cuda_cmd=utils/run.pl

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1
. ./cmd.sh || exit 1

if [ $# -ne 0 ]; then
    echo "Usage: $0 --src_dir <src_dir> --tgt_dir <tgt_dir> --file_name wav.scp --codec_choice DAC"
    exit 0
fi

# TODO (Jinchuan): check the installation of the used codec models

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [[ ${file_name} == *.scp ]]; then
        file_name="${file_name%.scp}"
    else
        echo "file_name should end with .scp suffix. ${file_name}"
    fi

    output_dir=${tgt_dir}/data
    mkdir -p "${output_dir}"
    _logdir=${tgt_dir}/logdir
    mkdir -p "${_logdir}"
    mkdir -p ${tgt_dir}/token_lists/

    nutt=$(<"${src_dir}"/${file_name}.scp wc -l)
    _nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${tgt_dir}/logdir/${file_name}.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl ${src_dir}/${file_name}.scp ${split_scps} || exit 1;

    wav_wspecifier="ark,scp:${output_dir}/${file_name}_resyn_${codec_choice}.JOB.ark,${output_dir}/${file_name}_resyn_${codec_choice}.JOB.scp"
    code_wspecifier="ark,scp:${output_dir}/${file_name}_codec_${codec_choice}.JOB.ark,${output_dir}/${file_name}_codec_${codec_choice}.JOB.scp"
    ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/codec_dump_${codec_choice}.JOB.log \
        ${python} pyscripts/feats/dump_codec.py \
            --codec_choice ${codec_choice} \
            --codec_fs ${codec_fs} \
            --batch_size ${batch_size} \
            --bias ${bias} \
            --dump_audio ${dump_audio} \
            --rank JOB \
            --vocab_file ${tgt_dir}/token_lists/codec_token_list \
            --wav_wspecifier ${wav_wspecifier} \
            --checkpoint_path ${checkpoint_path} \
            --config_path ${config_path} \
            "scp:${_logdir}/${file_name}.JOB.scp" ${code_wspecifier} || exit 1;

    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_codec_${codec_choice}.${n}.scp || exit 1;
    done > ${tgt_dir}/${file_name}.scp || exit 1

    if ${dump_audio}; then
        for n in $(seq ${_nj}); do
            cat ${output_dir}/${file_name}_resyn_${codec_choice}.${n}.scp || exit 1;
        done > ${tgt_dir}/${file_name}_resyn_${codec_choice}.scp || exit 1
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
