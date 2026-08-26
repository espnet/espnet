#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


train_set=
dev_set=
test_sets=
datadir=dump/raw/org
feat_dir=dump_feats
use_gpu=true

feature_type=hubert  # "mfcc", "hubert", "s3prl"
layer=12

# HuBERT related
hubert_type="fairseq"
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"

# s3prl related
s3prl_upstream_name=

nj=1
python=python3       # Specify python to execute espnet commands.

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--feature_type:mfcc>"
    exit 0
fi

if [ "${feature_type}" = "hubert" ] || [ "${feature_type}" = "s3prl" ]; then
    suffix=""
else
    suffix=""
    use_gpu=false  # mfcc feature does not require GPU.
fi


if ${use_gpu}; then
    _cmd="${cuda_cmd}"
    _ngpu=1
else
    _cmd="${train_cmd}"
    _ngpu=0
fi

for dset in "${train_set}" "${dev_set}" ${test_sets}; do
    echo "${dset}"

    # 1. Split the key file
    output_dir="${feat_dir}/${feature_type}/${suffix}${dset}/data"
    mkdir -p "${output_dir}"
    _logdir="${feat_dir}/${feature_type}/${suffix}${dset}/logdir"
    mkdir -p "${_logdir}"
    nutt=$(<"${datadir}/${dset}"/wav.scp wc -l)
    _nj=$((nj<nutt?nj:nutt))

    key_file="${datadir}/${dset}"/wav.scp
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/wav.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # shellcheck disableSC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/dump_feats.JOB.log \
        ${python} local/dump_feats.py \
            --in_filetype "sound" \
            --out_filetype "mat" \
            --feature_type "${feature_type}" \
            --hubert_type "${hubert_type}" \
            --hubert-model-url "${hubert_url}" \
            --hubert-model-path "${hubert_dir_path}" \
            ${s3prl_upstream_name:+--s3prl-upstream-name ${s3prl_upstream_name}} \
            --layer "${layer}" \
            --write_num_frames "ark,t:${_logdir}/utt2num_frames.JOB" \
            "scp:${_logdir}/wav.JOB.scp" \
            "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

    # concatenate scp files
    for n in $(seq ${_nj}); do
        cat ${output_dir}/feats.${n}.scp || exit 1;
    done > ${output_dir}/../feats.scp || exit 1

    for n in $(seq ${_nj}); do
        cat ${_logdir}/utt2num_frames.$n || exit 1;
    done > ${output_dir}/../utt2num_frames || exit 1
    rm ${_logdir}/utt2num_frames.*

done
