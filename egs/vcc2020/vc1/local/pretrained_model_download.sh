#!/bin/bash -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
pretrained_model=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <pretrained_model>"
    echo ""
    echo "Available pretrained models:"
    echo "    - tts1_en_de"
    echo "    - tts1_en_fi"
    echo "    - tts1_en_zh"
    echo "    - tts1 (LibriTTS)"
    echo "    - pwg_task1"
    echo "    - pwg_task2"
    exit 1
fi

case "${pretrained_model}" in
    "tts1_en_de") share_url="https://drive.google.com/open?id=1UvtFkqdkE8bOCKWXlEltc746JsCKaTMX" ;;
    "tts1_en_fi") share_url="https://drive.google.com/open?id=1XYpBZe9-9AgAxGpKfrgQPDjlW2S6duac" ;;
    "tts1_en_zh") share_url="https://drive.google.com/open?id=1E6vzNaXT6r7Zybefat_p9ncnMOQCXtem" ;;
    "tts1")       share_url="https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3" ;;
    "pwg_task1")  share_url="https://drive.google.com/open?id=" ;;
    "pwg_task2")  share_url="https://drive.google.com/open?id=1li9DLZGnAheWZrB4oXGo0KWq-fHuFH_l" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}
mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished donwload of pretrained model."
