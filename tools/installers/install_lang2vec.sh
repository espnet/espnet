#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1
fi

# Used by recipe target preparation, not by ESPnet library code or inference.
python3 -m pip install lang2vec==1.1.2
