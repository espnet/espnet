#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
pretrained_model=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <pretrained_model>"
    echo ""
    echo "Available pretrained models:"
    echo "    - tts1 (LibriTTS)"
    echo "    - tts1_en_de"
    echo "    - tts1_en_fi"
    echo "    - tts1_en_zh"
    echo "    - tts1_TEF1"
    echo "    - tts1_TEF2"
    echo "    - tts1_TEM1"
    echo "    - tts1_TEM2"
    echo "    - tts1_en_de_TGF1"
    echo "    - tts1_en_de_TGM1"
    echo "    - tts1_en_fi_TFF1"
    echo "    - tts1_en_fi_TFM1"
    echo "    - tts1_en_zh_TMF1"
    echo "    - tts1_en_zh_TMM1"
    echo "    - pwg_task1"
    echo "    - pwg_task2"
    exit 1
fi

case "${pretrained_model}" in
    "tts1")             share_url="https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3" ;;
    "tts1_en_de")       share_url="https://drive.google.com/open?id=1UvtFkqdkE8bOCKWXlEltc746JsCKaTMX" ;;
    "tts1_en_fi")       share_url="https://drive.google.com/open?id=1XYpBZe9-9AgAxGpKfrgQPDjlW2S6duac" ;;
    "tts1_en_zh")       share_url="https://drive.google.com/open?id=1E6vzNaXT6r7Zybefat_p9ncnMOQCXtem" ;;
    "tts1_TEF1")        share_url="https://drive.google.com/open?id=11qOvuMGP76BEe_pcPgYdqnWi05MIIYTA" ;;
    "tts1_TEF2")        share_url="https://drive.google.com/open?id=1y6IFgLMatjh9wspwu-oBba-rPOH1zKlS" ;;
    "tts1_TEM1")        share_url="https://drive.google.com/open?id=16Q3XOAfI5tG0LZ0SIKE166N3RCtzO722" ;;
    "tts1_TEM2")        share_url="https://drive.google.com/open?id=1EET2qhBi6nl0DH7UEg0Ez9SfgDX-cFM-" ;;
    "tts1_en_de_TGF1")  share_url="https://drive.google.com/open?id=1Vd4Qa8Dm9UQ-LZbyNPRiqoSgOZkGQsQi" ;;
    "tts1_en_de_TGM1")  share_url="https://drive.google.com/open?id=1bvyMfA-zKfO2LEogq-QXhHQeETxdBU29" ;;
    "tts1_en_fi_TFF1")  share_url="https://drive.google.com/open?id=1rA9ucA-VvhWkcFsGG6izBt2USOZY1_g6" ;;
    "tts1_en_fi_TFM1")  share_url="https://drive.google.com/open?id=1QfqwnTK0BKO0z_eYqltzL_MeqVGrMiZg" ;;
    "tts1_en_zh_TMF1")  share_url="https://drive.google.com/open?id=1kWBYSkvaQ0-7CwOfjVaWQYF0vEm0rNyS" ;;
    "tts1_en_zh_TMM1")  share_url="https://drive.google.com/open?id=13xDOSo53BSQoF1kD27SdwXoAGqtjtIEM" ;;
    "pwg_task1")        share_url="https://drive.google.com/open?id=11KKux-du6fvsMMB4jNk9YH23YUJjRcDV" ;;
    "pwg_task2")        share_url="https://drive.google.com/open?id=1li9DLZGnAheWZrB4oXGo0KWq-fHuFH_l" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}
mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished download of pretrained model."
