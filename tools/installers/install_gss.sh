#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# check if cupy is installed, if not install it with conda
if ! command -v gss &>/dev/null; then
  conda install -yc conda-forge cupy=10.2
  # then install gpu-gss
  pip install git+http://github.com/desh2608/gss
fi
