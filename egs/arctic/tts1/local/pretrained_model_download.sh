#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
pretrained_model=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <pretrained_model>"
    echo ""
    echo "Available pretrained models:"
    echo "    - mailabs.en_UK.elizabeth.transformer.v1.single"
    echo "    - mailabs.en_UK.elizabeth.tacotron2.v3"
    echo "    - mailabs.en_US.judy.transformer.v1.single"
    echo "    - mailabs.en_US.judy.tacotron2.v3"
    echo "    - mailabs.en_US.elliot.transformer.v1.single"
    exit 1
fi

case "${pretrained_model}" in
    "mailabs.en_UK.elizabeth.transformer.v1.single") share_url="https://drive.google.com/open?id=1iXdQv_YGD9VG1dR_xCjSkX6A4HkrpTbF" ;;
    "mailabs.en_UK.elizabeth.tacotron2.v3") share_url="https://drive.google.com/open?id=1iOwvCx6wX5_qCmHZSX_vCd_ZYn-B5akh" ;;
    "mailabs.en_US.judy.transformer.v1.single") share_url="https://drive.google.com/open?id=1rHQMMjkSoiX3JX2e70MKUKSrxHGwhmRb" ;;
    "mailabs.en_US.judy.tacotron2.v3") share_url="https://drive.google.com/open?id=1cNrTa8Jxa3AYcap7jo0_RPBapiay3etG" ;;
    "mailabs.en_US.elliot.transformer.v1.single") share_url="https://drive.google.com/open?id=1zv9GwhhBW32a6RM5wHzjqRxkkv9IrXTL" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}
mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished download of pretrained model."
