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

src_dir=
tgt_dir=
file_name=
batch_size=5
stage=1
stop_stage=100
nj=4
python=python3
hf_model_tag=null
max_words=1000

log "$0 $*"
. utils/parse_options.sh

. ./cmd.sh
. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 --src_dir <src_dir> --tgt_dir <tgt_dir> --file_name text --hf_model_tag google/gemma-2b"
    exit 0
fi

model_name="${hf_model_tag//\//_}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    output_dir=${tgt_dir}/data
    mkdir -p "${output_dir}"
    _logdir=${tgt_dir}/logdir
    mkdir -p "${_logdir}"

    if [ ! -f ${src_dir}/${file_name} ]; then
        echo "source file ${src_dir}/${file_name} is missing. Exit" && exit 1;
    fi

    nutt=$(<"${src_dir}"/${file_name} wc -l)
    _nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${tgt_dir}/logdir/${file_name}.${n}"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl ${src_dir}/${file_name} ${split_scps} || exit 1;

    wspecifier="ark,scp:${output_dir}/${file_name}_text_emb_${model_name}.JOB.ark,${output_dir}/${file_name}_text_emb_${model_name}.JOB"
    ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/text_emb_dump_${model_name}.JOB.log \
        ${python} pyscripts/feats/dump_text_emb.py \
            --input_file ${tgt_dir}/logdir/${file_name}.${n} \
            --hf_model_tag ${hf_model_tag} \
            --max_words ${max_words} \
            --batch_size ${batch_size} \
            ${wspecifier} || exit 1;

    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_text_emb_${model_name}.${n} || exit 1;
    done > ${tgt_dir}/${file_name} || exit 1

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
