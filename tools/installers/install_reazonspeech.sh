#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

python3 -m pip install git+https://github.com/reazon-research/ReazonSpeech
