#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# The "<4.50.0" cap this used to add below torch 2.1.0 is gone with those
# versions: the oldest torch ESPnet supports is well past 2.1.0, so the branch
# could not be taken.
TR_VER="4.9.1"

python3 -m pip install "transformers>=${TR_VER}" soxr
