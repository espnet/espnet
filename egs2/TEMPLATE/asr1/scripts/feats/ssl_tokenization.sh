#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Speech tokenization by SSL model. This is a simplified version. For the version with 
# K-means model training, also check: scripts/feats/perform_kmeans.sh

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
file_name=
src_dir=
tgt_dir=
nj=4
fs=16000
batch_bins=4800000
use_gpu=true

python=python3
ssl_choice=espnet_hubert
checkpoint_path=null
kmeans_path=null
nlayer=16
hf_model_tag=null

log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 --src_dir <src_dir> --tgt_dir <tgt_dir> --file_name wav.scp"
    exit 0
fi

if [ "${hf_model_tag}" != "null" ]; then
    log "download from url is not supported yet" && exit 1;
elif [ ! -f ${checkpoint_path} ] || [ ! -f ${kmeans_path} ]; then
    log "SSL model or K-Means model is not available" && exit 1;
fi

if [ ${fs} != 16000 ]; then
    log "Currently only 16kHz model is supported" && exit 1;
fi

if [ ! -f ${tgt_dir}/utt2num_samples ]; then
    log "File ${tgt_dir}/utt2num_samples should also exist."
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

    for n in $(seq ${_nj}); do
        utils/filter_scp.pl ${tgt_dir}/logdir/${file_name}.${n}.scp ${tgt_dir}/utt2num_samples \
            > ${tgt_dir}/logdir/utt2num_samples.${n} &
    done; wait

    rspecifier="scp:${tgt_dir}/logdir/${file_name}.JOB.scp"
    wspecifier="ark,scp:${output_dir}/${file_name}_ssl_${ssl_choice}.JOB.ark,${output_dir}/${file_name}_ssl_${ssl_choice}.JOB.scp"
    feature_conf="{ \
        type=${ssl_choice}, \
        conf={ \
            sample_rate=${fs}, \
            hubert_model_path=${checkpoint_path}, \
            layer=${nlayer} \
        } \
    }"

    log "Start SSL tokenization. log in ${_logdir}/ssl_dump_${ssl_choice}.*.log"
    ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/ssl_dump_${ssl_choice}.JOB.log \
        ${python} pyscripts/feats/dump_km_label.py \
            --online_feature_extract true \
            --km_path "${kmeans_path}" \
            --batch_bins ${batch_bins} \
            --in_filetype "kaldi_ark" \
            --out_filetype "mat" \
            --use_gpu ${use_gpu} \
            --feature_conf "${feature_conf}" \
            --utt2num_samples ${tgt_dir}/logdir/utt2num_samples.JOB \
            ${rspecifier} ${wspecifier}


    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_ssl_${ssl_choice}.${n}.scp || exit 1;
    done > ${tgt_dir}/${file_name}.scp || exit 1

    n_clusters=$(python -c "import joblib; model = joblib.load('${kmeans_path}'); print(model.n_clusters)")
    for n in `seq ${n_clusters}`; do
        echo "<ssl_code${n}>"
    done > ${tgt_dir}/token_lists/ssl_token_list
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
