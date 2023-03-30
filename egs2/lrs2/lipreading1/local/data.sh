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

cmd=run.pl
nj=2
stage=1
stop_stage=5

log "$0 $*"
. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dataset in train val test; do
        mkdir -p data/${dataset}
        awk  -v lrs2=${LRS2} -F '[/ ]' '{print $1"_"$2, lrs2"/main/"$1"/"$2".mp4"}' ${LRS2}/${dataset}.txt | sort > data/${dataset}/video.scp
        awk '{print $1, "ffmpeg -i " $2 " -ar 16000 -ac 1  -f wav pipe:1 |" }' data/${dataset}/video.scp > data/${dataset}/wav.scp
        awk '{print $2}' data/${dataset}/video.scp | sed -e 's/.mp4/.txt/g' | while read -r line
        do
            grep 'Text:' $line | sed -e 's/Text:  //g'
        done > data/${dataset}/text_tmp
        paste  <(awk '{print $1}' data/${dataset}/wav.scp)  data/${dataset}/text_tmp >  data/${dataset}/text
        rm data/${dataset}/text_tmp
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/utt2spk
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/spk2utt

    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Download the pretrained model to extract visual features.
    # The model was trained by Chenda Li (lichenda1996@sjtu.edu.cn),
    # following the paper by Stafylakis, T., & Tzimiropoulos, G. (2017).
    # "Combining residual networks with LSTMs for lipreading".
    if [ ! -f ./local/feature_extract/lipread_lrw_pretrain.pt.tgz ]; then
        wget https://zenodo.org/record/5090353/files/lipread_lrw_pretrain.pt.tgz -O ./local/feature_extract/lipread_lrw_pretrain.pt.tgz
        tar xzvf ./local/feature_extract/lipread_lrw_pretrain.pt.tgz -C ./local/feature_extract/
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

  if python -c "import skvideo, skimage, face_alignment" &> /dev/null; then
    echo 'requirements installed'
  else
    echo 'please install required packages by run ". ./path.sh; pip install sk-video scikit-image face_alignment"'
    exit 1;
  fi

    for dataset in train val test; do
        echo "extracting visual feature for [${dataset}]"

        log_dir=data/${dataset}/split_${nj}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dataset}/video.scp $split_scps || exit 1

        $cmd JOB=1:$nj ${log_dir}/extract_visual_feature.JOB.log python ./local/feature_extract/extract_visual_feature.py \
         ${log_dir}/video.JOB.scp \
         scp,ark:${log_dir}/vfeature.JOB.scp,${log_dir}/vfeature.JOB.ark || exit 1

        for n in $(seq $nj); do
            cat ${log_dir}/vfeature.${n}.scp
        done > data/${dataset}/vfeature.scp
        cp data/${dataset}/vfeature.scp data/${dataset}/feats.scp

        ./steps/compute_cmvn_stats.sh data/${dataset}/ data/${dataset}/cmvn data/${dataset}
    done

fi
