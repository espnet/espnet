#!/bin/bash

set -e
set -u
set -o pipefail

if [ $# != 3 ]; then
    echo "Usage: $0 </path/to/AMI-MDM> <min duration> <sample rate>"
    echo "e.g. $0 /foo/bar/mdm8_train 0.1 16000"
    exit 1;
fi

dir=$1
min_dur=$2
sample_rate=$3

if [ ! -f ${dir}/utt2num_frames ]; then
    echo "Error: File ${data_dir}/utt2num_frames no found (run dump_pcm --write-utt2num-frames true)."
    exit 1
fi

awk '{utt=$1; duration=$2;if (duration>"'"$min_dur"'" * "'"$sample_rate"'") printf("%s %s\n", utt, duration)}' ${dir}/utt2num_frames | sort | uniq > ${dir}/utt2num_frames.tmp
cp ${dir}/utt2num_frames.tmp ${dir}/utt2num_frames

awk '{print $1}' ${dir}/utt2num_frames | sort -u | join - $dir/feats.scp | sort > $dir/feats.scp.tmp
cp ${dir}/feats.scp.tmp ${dir}/feats.scp

rm ${dir}/*.tmp
