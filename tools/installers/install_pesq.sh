#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi
unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi


if [ ! -e PESQ.zip ]; then
    wget --tries=3 --no-check-certificate \
        'https://github.com/LiChenda/itu_pesq/raw/main/T-REC-P.862-200511.zip' -O PESQ.zip
fi
if [ ! -e PESQ ]; then
    mkdir -p PESQ_P.862.2
    unzip PESQ.zip -d PESQ_P.862.2
    unzip "PESQ_P.862.2/Software/P862_annex_A_2005_CD  wav final.zip" -d PESQ_P.862.2
    rm -rf PESQ
    ln -s PESQ_P.862.2 PESQ
fi

(
    set -euo pipefail
    cd PESQ/P862_annex_A_2005_CD/source
    gcc ./*.c -lm -o PESQ
)
