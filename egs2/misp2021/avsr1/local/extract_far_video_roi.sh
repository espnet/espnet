#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Hang Chen)
# Apache 2.0

# extract region of interest (roi) in the video, store as npz file, item name is "data"

set -e
# configs for 'chain'
stage=0
nj=15
gpu_nj=4
# End configuration section.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


pip show -f opencv-python >/dev/null || pip install opencv-python

if [ $# != 3 ]; then
  echo "Usage: $0 <data-set> <roi-json-dir> <audio-dir>"
  echo " $0 data/train_far /path/roi data/train_far_sp_hires"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

data_set=$1
roi_json_dir=$2
audio_dir=$3
video_dir="$data_set"_video
av_dir="$audio_dir"_av
segment_video_dir="$video_dir"/segments_data
extractor_dir=local/extractor
visual_embedding_dir="$video_dir"/visual_embedding

###########################################################################
# prepare mp4.scp, segments, vid2spk, spk2vid, text
###########################################################################
if [ $stage -le 0 ]; then
  if [[ ! -f $video_dir/.done ]]; then
    mkdir -p $video_dir
    cat $data_set/mp4.scp > $video_dir/mp4.scp
    cat $data_set/segments > $video_dir/segments
    cat $data_set/spk2utt > $video_dir/spk2vid
    cat $data_set/utt2spk > $video_dir/vid2spk
    cat $data_set/text > $video_dir/text
    touch $video_dir/.done
  fi
fi

###########################################################################
# segment mp4 and crop roi, store as npz, item name is data
###########################################################################
if [ $stage -le 1 ]; then
  if [[ ! -f $video_dir/segment.done ]]; then
    mkdir -p $segment_video_dir/log
    for n in `seq $nj`; do
      cat <<-EOF > $segment_video_dir/log/roi.$n.sh
        python local/prepare_far_video_roi.py --ji $((n-1)) --nj $nj $video_dir $roi_json_dir $segment_video_dir
EOF
    done
    chmod a+x $segment_video_dir/log/roi.*.sh
    $train_cmd JOB=1:$nj $segment_video_dir/log/roi.JOB.log $segment_video_dir/log/roi.JOB.sh || exit 1;
    rm -f $segment_video_dir/log/roi.*.sh
    cat $segment_video_dir/log/roi.*.scp | sort -k 1 | uniq > $video_dir/roi.scp
    rm -f $segment_video_dir/log/roi.*.scp
    echo 'segment done'
    touch $video_dir/segment.done
  fi
fi

###########################################################################
# extract visual embedding
###########################################################################
if [ $stage -le 2 ]; then
  if [[ ! -f $video_dir/extract.done ]]; then
    mkdir -p $visual_embedding_dir/log
    #notice: -g 0 1 2 3 pay attention to your GPU numbers
    python local/prepare_visual_embedding_extractor.py $video_dir -nj $nj -g 0 1 2
    chmod a+x $visual_embedding_dir/log/extract.*.sh
    $train_cmd JOB=1:$nj $visual_embedding_dir/log/extract.JOB.log $visual_embedding_dir/log/extract.JOB.sh
    rm -f $visual_embedding_dir/log/extract.*.sh
    echo 'extract done'
    touch $video_dir/extract.done
  fi
fi


###########################################################################
# concatenate audio-visual embedding
###########################################################################
if [ $stage -le 3 ]; then
  if [ ! -f $video_dir/concatenate.done ]; then
    mkdir -p $av_dir/data
    mkdir -p $av_dir/log

    cat $data_set/segments > $av_dir/segments
    cat $data_set/spk2utt > $av_dir/spk2utt
    cat $data_set/utt2spk > $av_dir/utt2spk
    cat $data_set/text > $av_dir/text
    cat $data_set/wav.scp > $av_dir/wav.scp
    cat $video_dir/mp4.scp > $av_dir/mp4.scp

    python local/concatenate_feature.py --ji 0 --nj 1 $audio_dir $video_dir $av_dir/data

    cat $av_dir/data/raw_av_embedding.*.scp | sort -k 1 | uniq > $av_dir/feats.scp
    steps/compute_cmvn_stats.sh $av_dir || exit 1;
    utils/fix_data_dir.sh $av_dir || exit 1;
    echo 'concatenate done'
    touch $video_dir/concatenate.done
  fi
fi
