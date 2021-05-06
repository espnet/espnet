#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi


rm -rf py3mmseg
git clone https://github.com/kamo-naoyuki/py3mmseg
python3 -m pip install -e py3mmseg
