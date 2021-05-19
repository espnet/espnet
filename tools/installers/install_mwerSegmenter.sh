#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

wget --tries=3 https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar zxvf mwerSegmenter.tar.gz
rm mwerSegmenter.tar.gz

patch mwerSegmenter/hyp2sgm.py < installers/patch_mwerSegmenter/hyp2sgm.patch
patch mwerSegmenter/sgm2mref.py < installers/patch_mwerSegmenter/sgm2mref.patch
