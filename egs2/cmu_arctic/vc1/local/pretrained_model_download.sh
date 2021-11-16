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
    echo "    - m_ailabs.judy.vtn.tts_pt"
    echo "    - pwg_slt"
    echo "    - pwg_rms"
    exit 1
fi

case "${pretrained_model}" in
    "m_ailabs.judy.vtn_tts_pt")     share_url="https://drive.google.com/open?id=1mPf-BxX3t_pqFFV6MGPBRePm5kgNR5sM" ;;
    "m_ailabs.judy.taco2_tts_pt")   share_url="https://drive.google.com/open?id=1fRLw6EA0x55xa449i_YRjCgm8sgv3hJI" ;;
    "pwg_slt")                      share_url="https://drive.google.com/open?id=1v70TtwfmYtTHq9LvksX907mNTEv1G-J1" ;;
    "pwg_rms")                      share_url="https://drive.google.com/open?id=1ty_de85SNldzVJSMQrHwl1ASBdGdSRav" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}
mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} "tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished download of pretrained model."
