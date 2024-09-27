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

SECONDS=0

model_conf=
cmd=run.pl
nj=8  # number of GPUs to extract features
stage=4
stop_stage=5

# task 
task=visualtts
data_name=lrs2

# dataset
train_set=
valid_set=
test_sets=

# path
video_source_dir=/nfs-02/yuyue/visualtts/dataset/lrs2/video_25fps
text_source_dir=/nfs-02/yuyue/visualtts/dataset/lrs2/wav_16k
data_id_dir=/nfs-02/yuyue/visualtts/reference_code/espnet/egs2/lrs2/avhubert/data/all.txt

# get landmarks
nshard=8

# kmeans
portion=1.0

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./db.sh || exit 1;

_dsets="${train_set} ${valid_set} ${test_sets}"

# stage1: get data(change)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate doc  to be completed: download lrs2 dataset and process
    python ./local/prepare_data.py \
        --wav_source_dir '/nfs-02/yuyue/visualtts/dataset/lrs2/wav_16k' \
        --video_source_dir '/nfs-02/yuyue/visualtts/dataset/lrs2/video_25fps' \
        --data_split_dir '/nfs-02/yuyue/visualtts/dataset/lrs2/data_split'
fi

# stage2: download avhubert pretrain model
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Download the pretrained AV-HuBERT model to extract visual features from official AV-HuBERT repository.
    # https://facebookresearch.github.io/av_hubert/
    mkdir -p local/pretrained
    if [ ! -f local/pretrained/${model_conf}_vox_iter5.pt ]; then
        echo "Download pretrained model noise-pretrain/${model_conf}_vox_iter5.pt from https://facebookresearch.github.io/av_hubert/"
        echo "If the download continues to fail, download it manually from the site."

        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/${model_conf}_vox_iter5.pt -O local/pretrained/${model_conf}_vox_iter5.pt
    fi
fi  # https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt

# stage3: generate landmarks
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # step2: download necessary models
    echo "downloading necessary models for landmark detection..."
    mkdir -p local/pretrained
    if [ ! -f local/pretrained/cnn_detector.dat ]; then
        echo "Download cnn_detector from http://dlib.net/files/mmod_human_face_detector.dat.bz2"
        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 http://dlib.net/files/mmod_human_face_detector.dat.bz2 -O local/pretrained/cnn_detector.dat.bz2
        bunzip2 local/pretrained/cnn_detector.dat.bz2
    fi
    
    if [ ! -f local/pretrained/face_predictor.dat ]; then
        echo "Download face_predictor from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 http://dlib.net/files/mmod_human_face_detector.dat.bz2 -O local/pretrained/face_predictor.dat.bz2
        bunzip2 local/pretrained/face_predictor.dat.bz2
    fi
    
    for dset in ${_dsets}; do
        log_dir="data/${dset}/logs"
        mkdir -p ${log_dir}
        if [ ! -e data/${dset}/landmarks.scp ]; then
            
            echo "detecting landmarks..."
            ${cmd} RANK=1:$nshard ${log_dir}/detect_landmarks.RANK.log python ./local/detect_landmarks.py \
                --video_source_dir "data/${dset}/video.scp" \
                --landmark_dir "${log_dir}/landmarks" \
                --cnn_detector "local/pretrained/cnn_detector.dat" \
                --face_predictor "local/pretrained/face_predictor.dat" \
                --rank RANK \
                --nshard ${nshard} || exit 1
        fi

    done
fi

# stage4: generate avhubert feature
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Extract audio visual features
    echo "We use AV-HuBERT ${model_conf} configuration to extract features"

    # 这里指定需要额外安装的video相关包，可以修正一下
    if python -c "import skvideo, skimage, cv2, python_speech_features" &> /dev/null; then
        echo "requirements installed"
    else
        echo "please install required packages by run 'cd ../../../tools; source activate_python.sh; installers/install_visual.sh;'"
        exit 1;
    fi

    for dset in ${_dsets}; do
        echo "extracting visual feature for [${dset}]"
        log_dir="data/${dset}/logs"
        mkdir -p ${log_dir}
        output_dir="data/${dset}/data"
        mkdir -p ${output_dir}

        if [ -e data/${dset}/feats.scp ]; then  # finally get
            continue
        fi

        # split scps
        split_scps=""
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dset}/video.scp $split_scps || exit 1

        ${cmd} JOB=1:$nj ${log_dir}/extract_av_feature.JOB.log python ./local/extract_av_feature.py \
            --video_source_dir ${log_dir}/video.JOB.scp \
            --landmark_dir data/${dset}/logs/landmarks \
            --model ${model_conf} \
            --gpu JOB \
            --write_num_frames ark,t:${output_dir}/num_frames.JOB.txt \
            ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp || exit 1

        for n in $(seq $nj); do
            cat ${output_dir}/feats.${n}.scp
        done > data/${dset}/feats.scp

        for n in $(seq $nj); do
            cat ${output_dir}/num_frames.${n}.txt
        done > data/${dset}/num_frames.txt
    done

    for dset in ${_dsets}; do
        utils/fix_data_dir.sh data/${dset}
    done
    echo "finish extracting avhubert features"
fi

# stage5: learn kmeans for avhubert feature
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    km_dir="local/pretrained/km_dir"
    mkdir -p ${km_dir}
    km_log_dir="${km_dir}/logs"
    mkdir -p ${km_log_dir}

    if [ ! -f ${km_dir}/avhubert_km500.bin ]; then
        # get train dataset
        if (( $(echo "${portion} >= 1.0" | bc -l) )); then
            cp "data/train/feats.scp" "${km_dir}/train.scp"
        else
            nutt=$(<"${avhubert_dir}/train"/feats.scp wc -l)
            portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2)+1)}')

            utils/subset_scp.pl \
                ${portion_nutt} data/train/feats.scp \
                > "${km_dir}/train.scp" || exit 1;
            log "Subsampling ${portion_nutt} utterances for Kmeans training."
        fi

        # learn kmeans
        ${cmd} ${km_log_dir}/sklearn_km.log python ./pyscripts/utils/sklearn_km.py \
            --feats-dir "scp:${km_dir}/train.scp" \
            --feature-type avhubert \
            --portion ${portion} \
            --km-path ${km_dir}/avhubert_km500.bin || exit 1
        fi
fi

    # generate kmeans labels: avhubert_tokenization

        


