#!/usr/bin/env bash
set -e

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
pretrained_model=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <pretrained_model>"
    echo ""
    echo "Available pretrained models:"
    echo "    - jsut.phn.transformer"
    echo "    - jsut.phn.tacotron2"
    exit 1
fi

case "${pretrained_model}" in
    "jsut.24k.phn.transformer") share_url="https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD" ;;
    "jsut.24k.phn.tacotron2") share_url="https://drive.google.com/open?id=1kp5M4VvmagDmYckFJa78WGqh1drb_P9t" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}
mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished download of pretrained model."
