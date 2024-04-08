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

wget --no-check-certificate --tries=3 https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar zxvf mwerSegmenter.tar.gz
rm mwerSegmenter.tar.gz

patch mwerSegmenter/hyp2sgm.py < installers/patch_mwerSegmenter/hyp2sgm.patch
patch mwerSegmenter/sgm2mref.py < installers/patch_mwerSegmenter/sgm2mref.patch
