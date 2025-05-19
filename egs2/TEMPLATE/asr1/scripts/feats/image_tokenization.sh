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
nproc=16            # number of processes when loading and pre-processing images
python=python3      # Specify python to execute espnet commands.
model_choice=cosmos # cosmos or vila-u
model_tag=Cosmos-0.1-Tokenizer-DI16x16
resize_choice=center_crop  # center_crop or border
batch_size=3
num_workers=16
resolution=256
file_name=image.scp
src_dir=
tgt_dir=
checkpoint_path=null
config_path=null
hf_model_tag=null

log "$0 $*"
. utils/parse_options.sh

# . ./path.sh || exit 1
. ./cmd.sh || exit 1

if [ $# -ne 0 ]; then
    echo "Usage: $0 --src_dir <src_dir> --tgt_dir <tgt_dir> --file_name image.scp"
    exit 0
fi

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

    image_wspecifier="ark,scp:${output_dir}/${file_name}_image_${model_choice}.JOB.ark,${output_dir}/${file_name}_image_${model_choice}.JOB.scp"
    ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/image_dump_${model_choice}.JOB.log \
        ${python} pyscripts/feats/dump_image.py \
            --model_choice ${model_choice} \
            --model_tag ${model_tag} \
            --resolution ${resolution} \
            --resize_choice ${resize_choice} \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --rank JOB \
            --vocab_file ${tgt_dir}/token_lists/image_token_list \
            "${_logdir}/${file_name}.JOB.scp" ${image_wspecifier} || exit 1;

    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_image_${model_choice}.${n}.scp || exit 1;
    done > ${tgt_dir}/${file_name}.scp || exit 1

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
