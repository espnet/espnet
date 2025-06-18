#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# adapted from https://github.com/simpleoier/ESPnet_SSL_ASR_tutorial_misc/blob/main/dump_ssl_feature.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


train_set=train
dev_set=dev
test_sets="test_doreco test_mswc test_fleurs"
datadir=dump/raw
feat_dir=dump_feats
use_gpu=true
feature_type=xeus
suffix=""

layer=-2
nj=32
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
    nutt=$(<"${datadir}${_suf}/${dset}"/wav.scp wc -l)
    _nj=$((nj<nutt?nj:nutt))

    key_file="${datadir}${_suf}/${dset}"/wav.scp
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/wav.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # also split utt2num_samples by JOB
    key_file="${datadir}${_suf}/${dset}"/utt2num_samples
    split_utt2num_samples=""
    for n in $(seq ${_nj}); do
        split_utt2num_samples+=" ${_logdir}/utt2num_samples.${n}"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_utt2num_samples}


    # shellcheck disableSC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/dump_feats.JOB.log \
        ${python} local/dump_xeus_feats.py \
            --in_filetype "sound" \
            --out_filetype "mat" \
            --layer "${layer}" \
            --write_num_frames "ark,t:${_logdir}/utt2num_frames.JOB" \
            --utt2num_samples "${_logdir}/utt2num_samples.JOB" \
            --batch_bins 1000000 \
            "scp:${_logdir}/wav.JOB.scp" \
            "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

    # concatenate scp files
    for n in $(seq ${_nj}); do
        cat ${output_dir}/feats.${n}.scp || exit 1;
    done > ${output_dir}/../feats.scp || exit 1

    for n in $(seq ${_nj}); do
        cat ${_logdir}/utt2num_frames.$n || exit 1;
    done > ${output_dir}/../utt2num_frames || exit 1

    # copy the feats.scp to data/*
    cp ${output_dir}/../feats.scp "data/${dset}"

    # sort the utterances by filename (each job above sorted by duration of utterance)
    utils/fix_data_dir.sh "data/${dset}"
done
