#!/bin/bash
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
stop_stage=2
feat_name=feats
file_name=
src_dir=
tgt_dir=
nj=4  # number of parallel jobs
batch_bins=4800000
use_gpu=true

python=python3  # Specify python to execute espnet commands.
ssl_choice=avhubert
checkpoint_path=null
kmeans_path=local/pretrained/km_dir/avhubert_km500.bin


# cpu_cmd="run.pl"
# cuda_cmd="run.pl"   

# skip_train_kmeans=false     # Whether to skip the kmeans model training
# nclusters=500       # Number of clusters of kmeans model
# portion=0.1         # Portion of data from training set used to train kmeans model

# feature_type=avhubert


log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi

if [ ! -f ${kmeans_path} ]; then
    log "SSL model or K-Means model is not available" && exit 1;
fi

if [ ! -f ${tgt_dir}/utt2num_frames ]; then
    log "File ${tgt_dir}/utt2num_frames should also exist."
fi
# input avhubert features, output kmeans labels 25fps video
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [[ ${file_name} == *.scp ]]; then  # feat.scp
        file_name="${file_name%.scp}"
    else
        echo "file_name should end with .scp suffix. ${file_name}"
    fi

    output_dir=${tgt_dir}/data  # data_feat/train/data
    mkdir -p "${output_dir}"
    _logdir=${tgt_dir}/logdir
    mkdir -p "${_logdir}"
    mkdir -p ${tgt_dir}/token_lists/

    nutt=$(<"${src_dir}"/${file_name}.scp wc -l)  # data_video/avhubert_dir/train
    _nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${tgt_dir}/logdir/${file_name}.${n}.scp"  # video.scp
    done
    # shellcheck disable=SC2086 cut to tgt_dir
    utils/split_scp.pl ${src_dir}/${file_name}.scp ${split_scps} || exit 1;

    for n in $(seq ${_nj}); do
        utils/filter_scp.pl ${tgt_dir}/logdir/${file_name}.${n}.scp ${tgt_dir}/utt2num_frames \
            > ${tgt_dir}/logdir/utt2num_frames.${n} &
    done; wait

    rspecifier="scp:${tgt_dir}/logdir/${file_name}.JOB.scp"
    wspecifier="ark,scp:${output_dir}/${file_name}_ssl_${ssl_choice}.JOB.ark,${output_dir}/${file_name}_ssl_${ssl_choice}.JOB.scp"
    feature_conf="{ \
        type=${ssl_choice} \
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
            --utt2num_samples ${tgt_dir}/logdir/utt2num_frames.JOB \
            ${rspecifier} ${wspecifier}


    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_ssl_${ssl_choice}.${n}.scp || exit 1;
    done > ${tgt_dir}/${file_name}.scp || exit 1

    n_clusters=$(python -c "import joblib; model = joblib.load('${kmeans_path}'); print(model.n_clusters)")
    for n in `seq ${n_clusters}`; do
        echo "<video_ssl_code${n}>"
    done > ${tgt_dir}/token_lists/video_ssl_token_list
fi

log "Successfully finished. [elapsed=${SECONDS}s]"



    

