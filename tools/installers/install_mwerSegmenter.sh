#!/bin/bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

wget --tries=3 https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar zxvf mwerSegmenter.tar.gz
rm mwerSegmenter.tar.gz
