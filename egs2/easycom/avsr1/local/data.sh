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

. ./db.sh
. ./path.sh
. ./cmd.sh

nj=1 # number of GPUs to extract features

stage=1
stop_stage=4

model_conf=$1
include_wearer=$2
with_LRS3=$3
noise_augmentation=$4

echo "#############################################"
echo "We use AV-HuBERT ${model_conf} configuration"
echo "Include wearer (Whether testing on the data of glasses wearer) is set as ${include_wearer}"
echo "LRS3 (Whether use data of LRS3) is set as ${with_LRS3}"
echo "Noise perturbation is set as ${noise_augmentation}"
echo "#############################################"

log "$0 $*"
. utils/parse_options.sh

if [ -z "${EASYCOM}" ]; then
    log "Fill the value of 'EASYCOM' of db.sh"
    log "Dataset can be download from https://github.com/facebookresearch/EasyComDataset"
    exit 1
fi

if [ ${with_LRS3} ]; then
    if [ -z "${LRS3}" ]; then
        log "Fill the value of 'LRS3' of db.sh"
        log "Dataset can be download from https://mmai.io/datasets/lip_reading/"
        exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Data preprocessing part. It consists the following things.
    # 1. Segmenting audio/video using ground truth diarization information.
    # 2. Landmark detection & Face crop.
    # 3. Text normalization.
    # We use the data split of the original paper
    # Data in "Session 4,12" are used for validation
    # Data in "Session 10,11" are used for test

    if python -c "import torchlm, skimage, cv2, python_speech_features, torchvision, av, gdown" &> /dev/null; then
        echo "requirements installed"
    else
        echo "please install required packages by run 'cd ../../../tools; source activate_python.sh; installers/install_visual.sh;'"
        exit 1;
    fi

    if [ ! -e data/preprocess.done ]; then
        python ./local/preprocessing.py --data_dir ${EASYCOM} --include_wearer ${include_wearer}
        touch data/preprocess.done
    fi

    if [ ${with_LRS3} ]; then
        if [ ! -e local/lrs3-valid.id ]; then
            # Download the validation file list from AV-HuBERT github.
            wget -O local/lrs3-valid.id https://raw.githubusercontent.com/facebookresearch/av_hubert/main/avhubert/preparation/data/lrs3-valid.id
        fi

        if [ ! -e local/LRS3_landmarks ]; then
            # Download extracted facial landmark to preprocess from the following repository.
            # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
            echo "Download extracted landmark from https://drive.google.com/uc?id=1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW"
            echo "If the download continues to fail, download it manually from the site & unzip at data/LRS3_landmarks."

            gdown https://drive.google.com/uc?id=1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW -O local/LRS3_landmarks.zip --continue
            unzip -qq local/LRS3_landmarks.zip
            rm local/LRS3_landmarks.zip
        fi

        if [ ! -e data/LRS3_preprocess.done ]; then
            python ./local/preprocessing.py --data_dir ${LRS3} --LRS3 --landmark local/LRS3_landmarks
            touch data/LRS3_preprocess.done
        fi
    fi

    if [ ${noise_augmentation} ]; then
        if [ ! -e local/babble_noise.wav ]; then
            # Download babble noise (NOISEX) from the following repository.
            # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
            wget -O local/babble_noise.wav https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/raw/master/pipelines/data/noise/babble_noise.wav
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ${with_LRS3} ]; then
        if [ ! -e data/test_with_LRS3/text ]; then
            python ./local/scp_gen.py --data_dir data/preprocess --include_wearer ${include_wearer} --LRS3 ${with_LRS3}
        fi
    else
        if [ ! -e data/test/text ]; then
            python ./local/scp_gen.py --data_dir data/preprocess --include_wearer ${include_wearer}
        fi
    fi

    for dataset in train val test; do
        if [ ${with_LRS3} ]; then
            dataset_name=${dataset}_with_LRS3
        else
            dataset_name=${dataset}
        fi

        if [ -e data/${dataset_name}/spk2utt ]; then
            continue
        fi
        cp data/${dataset_name}/wav.scp  data/${dataset_name}/video.scp
        awk '{print $1, $1}' data/${dataset_name}/wav.scp > data/${dataset_name}/utt2spk
        utils/utt2spk_to_spk2utt.pl data/${dataset_name}/utt2spk > data/${dataset_name}/spk2utt || exit 1;
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Download the pretrained AV-HuBERT model to extract visual features from official AV-HuBERT repository.
    # https://facebookresearch.github.io/av_hubert/
    mkdir -p local/pre-trained
    if [ ! -f local/pre-trained/${model_conf}_vox_iter5.pt ]; then
        echo "Download pre-trained model noise-pretrain/${model_conf}_vox_iter5.pt from https://facebookresearch.github.io/av_hubert/"
        echo "If the download continues to fail, download it manually from the site."

        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/${model_conf}_vox_iter5.pt -O local/pre-trained/${model_conf}_vox_iter5.pt
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Extract audio visual features and fuse the two features using pre-trained AV-HuBERT front-ends
    for dataset in train val test; do
        if [ ${with_LRS3} ]; then
            dataset_name=${dataset}_with_LRS3
        else
            dataset_name=${dataset}
        fi

        if [ -e data/${dataset_name}/feats.scp ]; then
            continue
        fi

        echo "extracting audio-visual feature for ${dataset_name}"
        log_dir=data/${dataset_name}/split_${nj}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dataset_name}/video.scp $split_scps || exit 1

        ${train_cmd} JOB=1:$nj ${log_dir}/extract_av_feature.JOB.log python ./local/extract_av_feature.py \
            --file_list ${log_dir}/video.JOB.scp \
            --model ${model_conf} \
            --gpu JOB \
            --write_num_frames ark,t:${log_dir}/num_frames.JOB.txt \
            --wspecifier ark,scp:${log_dir}/feature.JOB.ark,${log_dir}/feature.JOB.scp || exit 1

        for n in $(seq $nj); do
            cat ${log_dir}/feature.${n}.scp
        done > data/${dataset_name}/feats.scp

        for n in $(seq $nj); do
            cat ${log_dir}/num_frames.${n}.txt
        done > data/${dataset_name}/num_frames.txt
    done

    for dataset in train val test; do
        if [ ${with_LRS3} ]; then
            dataset_name="${dataset}_with_LRS3"
        else
            dataset_name="${dataset}"
        fi
        utils/fix_data_dir.sh data/${dataset_name}
    done

    if [ ${noise_augmentation} ]; then
        if [ ! -e data/babble_noise.pt ]; then
            python ./local/extract_av_feature.py \
            --model ${model_conf} \
            --noise_extraction \
            --noise_file local/babble_noise.wav \
            --noise_save_name data/babble_noise.pt
        fi
    fi
fi
