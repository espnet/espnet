#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf ParallelWaveGAN

# ParallelWaveGAN  Commit id when making this PR: `commit 4615144d75bcb519ff1d2df7699ddd626787b5a4`
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
cd ParallelWaveGAN
pip install "setuptools<80.0.0"
# --use-pep517: see install_warp-transducer.sh. Without it a setup.py-only
# project takes pip's legacy editable path, which setuptools>=80 re-invokes in a
# fresh isolated environment, and the `import torch` in that setup.py then fails.
pip install --use-pep517 --no-build-isolation -e .
cd ..
